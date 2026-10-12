import os
from ourUtils.torch.aes.conv_ae import CNNAutoencoderFlexible
from fastai.vision import conv_layer, res_block, conv2d_trans
import torch.utils.data
from torch import nn
import torchvision.transforms as transforms
from ourUtils.funct_band_aids import setup_directory
import numpy as np
from torchvision import datasets
from pretrain_stickfigures import pretrain
from torchvision.datasets.folder import pil_loader
from ourUtils.enrc_utils import random_seed, calc_nmi, run_enrc, get_embedding


def init_model(embd_sz, nfs, data_dir, random_state):
    tie_weights = False
    dropout_rate = 0.2
    # LeakyRelu
    slope = 0.1

    def conv_and_res(ni, nf, stride=2): return [conv_layer(
        ni, nf, stride=stride, leaky=slope), nn.Dropout2d(p=dropout_rate), res_block(nf, leaky=slope)]

    def conv_trans_and_res(ni, nf, last_layer, stride=2):
        if last_layer:
            return [conv2d_trans(ni, nf, stride=stride), nn.Dropout2d(p=dropout_rate), res_block(nf, leaky=1.0)]
        else:
            return [conv2d_trans(ni, nf, stride=stride), nn.Dropout2d(p=dropout_rate), res_block(nf, leaky=slope)]

    from fastai.vision import ImageImageList
    src = (ImageImageList.from_folder(path=data_dir, convert_mode='L')
                         .split_by_rand_pct(valid_pct=0.1, seed=random_state)
                         .label_from_func(func=lambda x: x, convert_mode='L'))
    data = src.databunch(bs=16)
    ae_model = CNNAutoencoderFlexible(data=data,
                                      nfs=nfs,
                                      layer_enc=conv_and_res,
                                      embd_sz=embd_sz,
                                      layer_dec=conv_trans_and_res,
                                      first_upscale=2,
                                      tie_weights=tie_weights,
                                      batch_norm_ae=False,
                                      dropout_ae=dropout_rate,
                                      )
    return ae_model


def load_model(model_dir, data_dir, nfs, embd_sz, seed):
    ae_model = init_model(embd_sz, nfs, data_dir,
                          random_state=seed)
    ae_dict = torch.load(os.path.join(model_dir))
    ae_state_dict = ae_dict["model"]
    ae_model.load_state_dict(ae_state_dict, strict=True)
    return ae_model


def get_pt_data(dl):
    pt_data = []
    for batch in dl:
        img, label = batch
        pt_data.append(img)
    pt_data = torch.cat(pt_data, dim=0)
    return pt_data


def main(np_seeds=None):
    nr_aes = 10
    if np_seeds is None:
        np_seeds = np.random.randint(100000, size=nr_aes)
    else:
        if np_seeds.shape[0] < nr_aes:
            raise ValueError(
                f"passed seeds {np_seeds.shape[0]} are smaller than number of aes {nr_aes}")
    bs = 32
    n_iterations = 2000
    cluster_lr = 1e-2
    pretrain_lr = cluster_lr / 4.0  # initial lr 1e-2
    # Dataset statistics
    data_dir = os.path.join('data', "stickfigures")
    std = (0.3332, )
    mean = (0.2585,)
    result_dir = os.path.join("enrc_results", "stickfigures")
    setup_directory(result_dir)

    def to_grey(x):
        return x.convert('L')

    transform = transforms.Compose([to_grey,
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean,
                                                         std),
                                    ])

    np_data = np.loadtxt(os.path.join(
        data_dir, 'stickfigures_3sub.data'), delimiter=";")

    np_labels = np_data[:, [0, 1, 2]].astype(np.int)
    trainset = datasets.DatasetFolder(
        root=data_dir, loader=pil_loader, extensions=(".png",), transform=transform)
    testloader = torch.utils.data.DataLoader(
        trainset, batch_size=bs, shuffle=False, drop_last=False)

    # match data and labels
    pt_data = get_pt_data(testloader)
    pt_labels_idx = [int(i[0].split("//")[-1].split("_")[-1].strip(".png"))
                     for i in trainset.samples]
    pt_labels = np_labels[pt_labels_idx, :]
    pt_labels = pt_labels[:, [0, 1]]

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

            model_dir = os.path.join(
                data_dir, "train", "models", model_name)

            ae_model = load_model(model_dir, data_dir,
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
                ae_model, cluster_layer, testloader, pt_labels, classes=["upper_body", "lower_body"])
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
