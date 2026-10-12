import os
from ourUtils.torch.aes.conv_ae import CNNAutoencoderFlexible
from fastai.vision import conv_layer, res_block, conv2d_trans
import torch.utils.data
from torch import nn
import torchvision.transforms as transforms
from ourUtils.funct_band_aids import setup_directory
import numpy as np
from pretrain_gtsrb import pretrain
from torchvision import datasets
from ourUtils.enrc_utils import random_seed, calc_nmi, run_enrc, get_embedding


def init_model(embd_sz, nfs, gtsrb_dir, random_state):
    dropout_rate = 0.1
    slope = 0.1
    tie_weights = False

    def conv_and_res(ni, nf, stride=2): return [conv_layer(
        ni, nf, stride=stride, leaky=slope), nn.Dropout2d(p=dropout_rate), res_block(nf, leaky=slope)]

    def conv_trans_and_res(ni, nf, last_layer, stride=2):
        if last_layer:
            return [conv2d_trans(ni, nf, stride=stride), nn.Dropout2d(p=dropout_rate), res_block(nf, leaky=1.0)]
        else:
            return [conv2d_trans(ni, nf, stride=stride), nn.Dropout2d(p=dropout_rate), res_block(nf, leaky=slope)]

    from fastai.vision import ImageImageList
    src = (ImageImageList.from_folder(path=gtsrb_dir)
           .split_by_rand_pct(valid_pct=0.2, seed=random_state)
           .label_from_func(func=lambda x: x))
    # bs doesn't matter only shape is important
    data = src.databunch(bs=16).normalize()
    ae_model = CNNAutoencoderFlexible(data=data,
                                      nfs=nfs,
                                      layer_enc=conv_and_res,
                                      embd_sz=embd_sz,
                                      layer_dec=conv_trans_and_res,
                                      first_upscale=2,
                                      tie_weights=tie_weights,
                                      batch_norm_ae=False,
                                      dropout_ae=dropout_rate)
    return ae_model


class CustomImageFolder(datasets.folder.DatasetFolder):
    from torchvision.datasets.folder import default_loader

    def __init__(self, root, extensions, transform=None, target_transform=None, loader=default_loader):
        super(CustomImageFolder, self).__init__(root, loader, extensions,
                                                transform=transform,
                                                target_transform=target_transform)
        self.imgs = self.samples


def load_model(model_dir, gtsrb_dir, nfs, embd_sz, seed):
    ae_model = init_model(embd_sz, nfs, gtsrb_dir,
                          random_state=seed)
    ae_dict = torch.load(os.path.join(model_dir))
    ae_state_dict = ae_dict["model"]
    ae_model.load_state_dict(ae_state_dict, strict=True)
    return ae_model


def clean_labels(labels):
    l = np.array(labels)
    print("Labels: ")
    for col_idx in range(l.shape[1]):
        label_list = l[:, col_idx].tolist()
        unique_labels = list(set(label_list))
        unique_labels.sort()
        print(unique_labels)
        mapping = {}
        for label_idx, label_i in enumerate(unique_labels):
            mapping[label_i] = label_idx

        l[:, col_idx] = [mapping[i] for i in label_list]
    l = l.astype(np.int)
    return l


def get_nr_labels(labels):
    labels_nr = []
    for l_i in labels:
        if l_i == 0:
            labels_nr.append(["03_speed_limit_70", "red_white_black"])
        elif l_i == 1:
            labels_nr.append(["09", "red_white_black"])
        elif l_i == 2:
            labels_nr.append(["35_ahead_only", "blue_white"])
        elif l_i == 3:
            labels_nr.append(["38_keep_right", "blue_white"])
    return labels_nr


def get_pt_data(dl):
    pt_data = []
    pt_labels = []
    for batch in dl:
        img, label = batch
        pt_data.append(img)
        pt_labels.append(label)
    pt_data = torch.cat(pt_data, dim=0).view((-1, 3, 32, 32))
    pt_labels = torch.cat(pt_labels)
    pt_labels = get_nr_labels(pt_labels)
    pt_labels = clean_labels(pt_labels)
    return pt_data, pt_labels


def main(np_seeds=None):
    nr_aes = 10
    if np_seeds is None:
        np_seeds = np.random.randint(100000, size=nr_aes)
    else:
        if np_seeds.shape[0] < nr_aes:
            raise ValueError(
                f"passed seeds {np_seeds.shape[0]} are smaller than number of aes {nr_aes}")

    bs = 64
    n_iterations = 20000
    cluster_lr = 1e-2
    pretrain_lr = cluster_lr / 4.0  # initial lr 1e-2/4.0

    # Dataset statistics
    gtsrb_dir = os.path.join('data', 'gtsrb')
    gtsrb_image_dir = os.path.join('data', 'gtsrb', 'train')

    result_dir = os.path.join("enrc_results", "gtsrb")
    setup_directory(result_dir)

    # Hist equalized
    std = (0.2920, 0.2910, 0.2926)
    mean = (0.5130, 0.5133, 0.5137)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean,
                                                         std),
                                    ])

    trainset = CustomImageFolder(root=gtsrb_image_dir, extensions=[
        ".png"], transform=transform)
    helperloader = torch.utils.data.DataLoader(
        trainset, batch_size=bs, shuffle=True, drop_last=False)
    pt_data, pt_labels = get_pt_data(helperloader)
    test_labels = pt_labels
    testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*(pt_data, torch.from_numpy(test_labels))), batch_size=bs,
                                             shuffle=False,
                                             drop_last=False)

    np.savetxt(os.path.join(result_dir, "labels.csv"),
               pt_labels.astype(np.int), delimiter=";")

    nfs = [128, 256]
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

            model_dir = os.path.join(
                gtsrb_dir, f"models/{model_name}")

            ae_model = load_model(model_dir, gtsrb_dir,
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
                ae_model, cluster_layer, testloader, pt_labels, classes=["type", "color"])
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
