import os
import sys
from fastai.basics import *
from fastai.vision import *
from fastai.callbacks import *
import fastai
import fastai.utils.collect_env
from ourUtils.torch.aes.conv_ae import CNNAutoencoderFlexible
from ourUtils.pretrain_utils import random_seed, finetune


def pretrain(np_seeds=None):
    nr_aes = 10
    if np_seeds is None:
        np_seeds = np.random.randint(100000, size=nr_aes)
    else:
        if np_seeds.shape[0] < nr_aes:
            raise ValueError(
                f"passed seeds {np_seeds.shape[0]} are smaller than number of aes {nr_aes}")

    bs = 64
    # L2 regularization
    wd = 1e-3
    finetune_iterations = 5000
    epochs = 106  # approx. 10000 iterations with bs of 64
    max_lr = 1e-2
    tie_weights = False

    # we prefer float32 format instead of float64 and to use of GPU if available
    USE_GPU = True
    dtype = torch.float32
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('using device:', device)
    src_path = os.path.join(
        'data', 'gtsrb')

    dropout_rate = 0.1
    slope = 0.1

    def conv_and_res(ni, nf, stride=2): return [conv_layer(
        ni, nf, stride=stride, leaky=slope), nn.Dropout2d(p=dropout_rate), res_block(nf, leaky=slope)]

    def conv_trans_and_res(ni, nf, last_layer, stride=2):
        if last_layer:
            return [conv2d_trans(ni, nf, stride=stride), nn.Dropout2d(p=dropout_rate), res_block(nf, leaky=1.0)]
        else:
            return [conv2d_trans(ni, nf, stride=stride), nn.Dropout2d(p=dropout_rate), res_block(nf, leaky=slope)]

    nfs = [128, 256]
    src = (ImageImageList.from_folder(path=src_path, extensions='.png')
           .split_none()
           .label_from_func(func=lambda x: x))
    stats = src.databunch(
        bs=len(src.databunch().label_list.train), device="cpu").batch_stats()
    for hl in [16]:
        embd_sz = hl

        def my_loss(preds, target):
            emb_enc, reconstruction = preds
            return F.mse_loss(reconstruction, target)

        for ae_index in range(nr_aes):
            print("\nStart training ae {} with random seed {}".format(
                ae_index, np_seeds[ae_index]))
            random_seed(np_seeds[ae_index])
            # flip and max_warp doesnt make sense here (flip changes meaning of the traffic sign and max_warp distorts the already distorted images even more)
            tfms = get_transforms(do_flip=False, max_warp=None)
            data = (src.transform(tfms, tfm_y=True)
                    .databunch(bs=bs)
                    .normalize(stats=stats, do_y=True))
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

            learn = Learner(data, ae_model, loss_func=my_loss, wd=wd)
            learn.fit_one_cycle(epochs, max_lr=max_lr, wd=wd)
            finetune(learn, finetune_iterations, lr=max_lr/2.0)
            learn.save(
                f'ae-model-hl-{embd_sz}-idx-{ae_index}')


if __name__ == "__main__":
    pretrain()
