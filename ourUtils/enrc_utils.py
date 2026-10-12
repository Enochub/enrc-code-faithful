import torch
import numpy as np
import pandas as pd
import random
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from collections import OrderedDict
from ourUtils.torch.aes import ae_utils
from enrc.enrc import Enrc


def random_seed(seed):
    random.seed(seed)
    np.random.seed(random.randint(0, 10000))
    torch.manual_seed(random.randint(0, 10000))
    torch.cuda.manual_seed_all(random.randint(0, 10000))
    torch.backends.cudnn.deterministic = True


def get_right_combination(clustering, labels):
    nmis = []
    for col_idx in range(labels.shape[1]):
        nmi_i = nmi_score(
            clustering, labels[:, col_idx], average_method="arithmetic")
        nmis.append(nmi_i)
    which = np.argmax(nmis)
    combi = (which, clustering, labels[:, which].copy())
    return combi


def nmi_combinations(clustering, labels):
    nmis = []
    for col_idx in range(clustering.shape[1]):
        nmi_i = nmi_score(
            clustering[:, col_idx], labels, average_method="arithmetic")
        nmis.append(nmi_i)
    return nmis


def calc_nmi(ae_model, cluster_layer, data_loader, labels, classes, device=torch.device("cuda")):
    ae_model.eval()
    ae_model.to(device)
    cluster_layer.to(device)
    cluster_layer.centers = [i.to(device) for i in cluster_layer.centers]
    cluster_layer.mask_sum = [i.to(device) for i in cluster_layer.mask_sum]
    result, z_rot = cluster_layer.predict_encode_batchwise(
        ae_model, data_loader, device=device)
    res_dict = OrderedDict({})
    for i, class_label in enumerate(classes):
        res_dict[class_label] = nmi_combinations(result, labels[:, i])

    df = pd.DataFrame(res_dict).round(2).T
    print("NMI:", df)
    torch.cuda.empty_cache()
    return df, result


def get_embedding(ae_model, testloader):
    encodings = []
    for batch in testloader:
        batch_data = batch[0].cuda()
        z = ae_model.encode(batch_data)
        encodings.append(z.detach().cpu())
    return torch.cat(encodings).numpy()


def train_one_round(ae_model, cluster_layer, batch, rec_loss_weight=1.0):
    loss_fn = torch.nn.MSELoss()
    z = cluster_layer.embedding_fn(batch)
    subspace_losses, z_rot_back = cluster_layer(z)
    reconstructed_rot = ae_model.decode(z_rot_back)
    rec_loss = loss_fn(reconstructed_rot, batch)
    summed_loss = subspace_losses + rec_loss_weight * rec_loss
    return summed_loss, subspace_losses, rec_loss


def run_enrc(ae_model, pt_data, pt_labels, pretrain_lr, cluster_lr, n_iterations, trainloader, random_state):
    clustered_spaces = []
    for labels_i in pt_labels.transpose():
        clustered_spaces.append(len(list(set(labels_i))))
    print("Used clustered spaces: ", clustered_spaces)

    ae_model.eval()
    ae_model.cuda()
    device = torch.device("cuda")
    bs = trainloader.batch_size
    print("Start init")

    def embedding_fn(X):
        if X.shape[0] > bs:
            data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*(X, X)),
                                                      batch_size=bs,
                                                      shuffle=False,
                                                      drop_last=False)
            return ae_utils.encode_batchwise(ae_model, data_loader, device=device)
        else:
            return ae_model.encode(X)
    
    cluster_layer = Enrc.run_init(org_space_data=pt_data,
                                  clusterings=clustered_spaces,
                                  embedding_fn=embedding_fn,
                                  rounds=100,
                                  random_state=random_state,
                                  max_iter=10)

    torch.cuda.empty_cache()
    cluster_layer.cuda()
    cluster_layer.centers = [i.cuda() for i in cluster_layer.centers]
    cluster_layer.mask_sum = [i.cuda() for i in cluster_layer.mask_sum]
    device = torch.device("cuda")
    param_dict = [{'params': ae_model.parameters(),
                   'lr': pretrain_lr},
                  {'params': [cluster_layer.rotation_layer.v],
                   'lr': pretrain_lr},
                  {'params': [cluster_layer.beta_weights],
                   'lr': cluster_lr},
                  ]
    optimizer = torch.optim.Adam(param_dict)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=2000, gamma=0.5)
    cluster_layer.center_lr = 0.5
    i = 0
    while(i < n_iterations):
        for batch in trainloader:
            # Deactivate Batchnorm and dropout
            ae_model.eval()
            batch = batch[0].cuda()
            summed_loss, subspace_losses, rec_loss = train_one_round(
                ae_model, cluster_layer, batch, rec_loss_weight=1.0)
            optimizer.zero_grad()
            summed_loss.backward()
            optimizer.step()
            ae_model.eval()
            changed = False
            for subspace_i in range(len(cluster_layer.centers)):
                changed_i = cluster_layer.reinit_centers(
                    subspace_i, n_samples=1000)
                if changed_i:
                    changed = True
            if changed:
                print("Reinit Threshold: ",
                      cluster_layer.reinit_threshold)
            if i % 50 == 0:
                ae_model.eval()
                with torch.no_grad():
                    # Rotation loss is calculated to check if it is not deviating to heavily
                    ident = torch.matmul(cluster_layer.rotation_layer.v.t(
                    ), cluster_layer.rotation_layer.v).detach().cpu()
                    rotation_loss = (
                        ident - torch.eye(n=ident.shape[0])).abs().mean()
                    print(f"step_i: {i} summed_loss: {summed_loss.item():.4f}, subspace_losses: {subspace_losses.item():.4f},rec_loss: {rec_loss.item():.4f},rotation_loss: {rotation_loss.item():.4f}")
            i += 1
            cluster_layer.reinit_threshold = int(np.sqrt(i))

            scheduler.step()
            if i == scheduler.step_size:
                cluster_layer.center_lr *= scheduler.gamma
            if i > n_iterations:
                break
        torch.cuda.empty_cache()

    return ae_model, cluster_layer
