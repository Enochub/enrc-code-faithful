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

    bs = 128
    # L2 regularization
    wd = 1e-3
    finetune_iterations = 5000
    epochs = 22  # approx. 10000 iterations with bs of 128
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
    src_path = os.path.join('data', 'cmnist')

    dropout_rate = 0.1
    slope = 0.1

    def conv_and_res(ni, nf, stride=2): return [conv_layer(
        ni, nf, stride=stride, leaky=slope), nn.Dropout2d(p=dropout_rate), res_block(nf, leaky=slope)]

    def conv_trans_and_res(ni, nf, last_layer, stride=2):
        if last_layer:
            return [conv2d_trans(ni, nf, stride=stride), nn.Dropout2d(p=dropout_rate), res_block(nf, leaky=1.0)]
        else:
            return [conv2d_trans(ni, nf, stride=stride), nn.Dropout2d(p=dropout_rate), res_block(nf, leaky=slope)]

    # Used to fit the convolution
    def crop_4pxls(x): return x[:, :, 4:-4, 4:-4]

    nfs = [64, 128]
    src = (ImageImageList.from_folder(path=src_path, convert_mode='L')
           .split_none()
           .label_from_func(func=lambda x: x, convert_mode='L'))
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
            # flip and lightning doesnt make sense here (have no light source and pictures are concatenated)
            tfms = get_transforms(do_flip=False, max_lighting=None)
            data = (src.transform(tfms, tfm_y=True)
                    .databunch(bs=bs)
                    .normalize(stats=stats, do_y=True))
            ae_model = CNNAutoencoderFlexible(data=data,
                                              nfs=nfs,
                                              layer_enc=conv_and_res,
                                              embd_sz=embd_sz,
                                              layer_dec=conv_trans_and_res,
                                              first_upscale=4,
                                              tie_weights=tie_weights,
                                              batch_norm_ae=False,
                                              dropout_ae=dropout_rate,
                                              output_fn=crop_4pxls,
                                              )

            learn = Learner(data, ae_model, loss_func=my_loss, wd=wd)
            learn.fit_one_cycle(epochs, max_lr=max_lr, wd=wd)
            finetune(learn, finetune_iterations, lr=max_lr/2.0)
            learn.save(
                f'ae-model-hl-{embd_sz}-idx-{ae_index}')


if __name__ == "__main__":
    pretrain()
