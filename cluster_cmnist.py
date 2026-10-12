import os
from ourUtils.torch.aes.conv_ae import CNNAutoencoderFlexible
from fastai.vision import conv_layer, res_block, conv2d_trans
import torch.utils.data
from torch import nn
import torchvision.transforms as transforms
from ourUtils.funct_band_aids import setup_directory
import numpy as np
from torchvision import datasets
from pretrain_cmnist import pretrain
from ourUtils.enrc_utils import random_seed, calc_nmi, run_enrc, get_embedding


def init_model(embd_sz, nfs, mnist_dir, random_state):
    tie_weights = False
    dropout_rate = 0.1
    # LeakyRelu
    slope = 0.1

    def conv_and_res(ni, nf, stride=2): return [conv_layer(
        ni, nf, stride=stride, leaky=slope), nn.Dropout2d(p=dropout_rate), res_block(nf, leaky=slope)]

    def conv_trans_and_res(ni, nf, last_layer, stride=2):
        if last_layer:
            return [conv2d_trans(ni, nf, stride=stride), nn.Dropout2d(p=dropout_rate), res_block(nf, leaky=1.0)]
        else:
            return [conv2d_trans(ni, nf, stride=stride), nn.Dropout2d(p=dropout_rate), res_block(nf, leaky=slope)]

    def crop_4pxls(x): return x[:, :, 4:-4, 4:-4]

    from fastai.vision import ImageImageList
    src = (ImageImageList.from_folder(path=mnist_dir, convert_mode='L')
                         .split_by_rand_pct(valid_pct=0.1, seed=random_state)
                         .label_from_func(func=lambda x: x, convert_mode='L'))
    data = src.databunch(bs=16)
    ae_model = CNNAutoencoderFlexible(data=data,
                                      nfs=nfs,
                                      layer_enc=conv_and_res,
                                      embd_sz=embd_sz,
                                      layer_dec=conv_trans_and_res,
                                      first_upscale=4,
                                      tie_weights=tie_weights,
                                      batch_norm_ae=False,
                                      output_fn=crop_4pxls,
                                      dropout_ae=dropout_rate,
                                      )
    return ae_model


def load_model(model_dir, mnist_dir, nfs, embd_sz, seed):
    ae_model = init_model(embd_sz, nfs, mnist_dir,
                          random_state=seed)
    ae_dict = torch.load(os.path.join(model_dir))
    ae_state_dict = ae_dict["model"]
    ae_model.load_state_dict(ae_state_dict, strict=True)
    return ae_model


def get_pt_data(dl, class_dict):
    pt_data = []
    pt_labels = []
    for batch in dl:
        img, label = batch
        pt_data.append(img)
        for label_i in label.tolist():
            cl = class_dict[label_i].split("_")
            cl = torch.tensor([int(cl[0]), int(cl[1])])
            pt_labels.append(cl)
    pt_data = torch.cat(pt_data, dim=0)
    pt_labels = torch.stack(pt_labels)
    return pt_data, pt_labels


def main(np_seeds=None):
    nr_aes = 10
    if np_seeds is None:
        np_seeds = np.random.randint(100000, size=nr_aes)
    else:
        if np_seeds.shape[0] < nr_aes:
            raise ValueError(
                f"passed seeds {np_seeds.shape[0]} are smaller than number of aes {nr_aes}")
    bs = 128
    n_iterations = 20000
    cluster_lr = 1e-2
    pretrain_lr = cluster_lr / 4.0  # initial lr 1e-2
    # Dataset statistics
    mnist_dir = os.path.join('data', 'cmnist')
    std = (0.2274,)
    mean = (0.0653,)
    result_dir = os.path.join("enrc_results", "cmnist")
    setup_directory(result_dir)

    def to_grey(x):
        return x.convert('L')

    transform = transforms.Compose([to_grey,
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean,
                                                         std),
                                    ])

    trainset = datasets.ImageFolder(root=mnist_dir, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=bs, shuffle=True)
    classes = trainset.classes
    class_dict = {i: c_i for i, c_i in enumerate(classes)}
    pt_data, pt_labels = get_pt_data(trainloader, class_dict=class_dict)
    testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*(pt_data, pt_labels)), batch_size=bs,
                                             shuffle=False,
                                             drop_last=False)
    pt_labels = pt_labels.detach().cpu().numpy()

    np.savetxt(os.path.join(result_dir, "labels.csv"),
               pt_labels.astype(np.int), delimiter=";")

    nfs = [64, 128]
    for hl in [16]:
        embd_sz = hl
        for ae_index in range(nr_aes):
            print("\nStart training ae {} with random seed {}".format(
                ae_index, np_seeds[ae_index]))
            random_seed(np_seeds[ae_index])
            model_name = f"ae-model-hl-{embd_sz}-idx-{ae_index}.pth"

            trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                                      shuffle=True,
                                                      drop_last=True)

            model_dir = os.path.join(mnist_dir, "models", model_name)

            ae_model = load_model(model_dir, mnist_dir,
                                  nfs, embd_sz, seed=np_seeds[ae_index])

            ae_model.cuda()
            np_emb = get_embedding(ae_model, testloader)
            np.savetxt(os.path.join(result_dir, model_name.split(
                ".")[0]+".csv"), np_emb, delimiter=";")

            ae_model, cluster_layer = run_enrc(ae_model=ae_model,
                                               pt_data=pt_data,
                                               pt_labels=pt_labels,
                                               pretrain_lr=pretrain_lr,
                                               cluster_lr=cluster_lr,
                                               n_iterations=n_iterations,
                                               trainloader=trainloader,
                                               random_state=np_seeds[ae_index])

            # Save Results
            res_df, pred_labels = calc_nmi(
                ae_model, cluster_layer, testloader, pt_labels, classes=["left", "right"])
            pred_path = os.path.join(
                result_dir, model_name.split(".")[0]+"_nmis.csv")
            res_df.to_csv(pred_path, sep=";", index=False)
            np.savetxt(os.path.join(result_dir, model_name.split(".")[0]+".labels"),
                       pred_labels.astype(np.int), delimiter=";")

            del ae_model
            del cluster_layer
            del trainloader
            torch.cuda.empty_cache()


if __name__ == "__main__":
    nr_aes = 10
    np_seeds = np.random.randint(100000, size=nr_aes)
    pretrain(np_seeds)
    main(np_seeds)
