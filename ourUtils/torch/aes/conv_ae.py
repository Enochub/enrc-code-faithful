from ourUtils.torch.aes.basis_layers import TiedLinear
from fastai.basics import *
from fastai.vision import *
from fastai.callbacks import *
import fastai

import math


def prev_pow_2(x): return 2**math.floor(math.log2(x))


class AutoencoderFlexible(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoencoderFlexible, self).__init__()
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def reconstruct(self, x):
        rec = x
        for submod_i in self.decoder:
            if isinstance(submod_i, TiedLinear):
                tied_weight = self.encoder[0].weight.data.clone().t()
                rec = submod_i(rec, tied_weight)
            else:
                rec = submod_i(rec)
        return rec

    def forward(self, x):
        encoding = self.encoder(x)
        reconstruction = self.reconstruct(encoding)
        return encoding, reconstruction


def lin_bn_drop(n_in: int, n_out: int,
                bn: bool = True,
                actn: Optional[nn.Module] = relu(inplace = True),
                p: float = 0.):
    "Sequence of linear (`n_in`,`n_out`) layers followed by `actn`, batchnorm (if `bn`) and dropout (with `p`)."
    if bn:
        bias = False
    else:
        bias = True
    layers = [nn.Linear(n_in, n_out, bias)]
    if bn:
        layers.append(nn.BatchNorm1d(n_out))
    if actn is not None:
        layers.append(actn)
    if p > 0:
        layers.append(nn.Dropout(p))
    return layers


def construct_tied_decoder(encoder, tie_weights=True):
    net = []
    for sub_mod_i in encoder:
        if isinstance(sub_mod_i, nn.Linear):
            tied_weights = sub_mod_i.weight.clone().t()
            n_in, n_out = tied_weights.shape[1], tied_weights.shape[0]
            if tie_weights:
                net.append(TiedLinear(
                    n_in, n_out, bias=sub_mod_i.bias is not None))
            else:
                net.append(
                    nn.Linear(n_in, n_out, bias=sub_mod_i.bias is not None))
        elif isinstance(sub_mod_i, nn.BatchNorm1d):
            net.append(nn.BatchNorm1d(n_out))
        elif isinstance(sub_mod_i, nn.LeakyReLU):
            net.append(relu(inplace=sub_mod_i.inplace,
                            leaky=sub_mod_i.negative_slope))
        elif isinstance(sub_mod_i, nn.ReLU):
            net.append(relu(inplace=sub_mod_i.inplace))
        elif isinstance(sub_mod_i, nn.Dropout):
            net.append(nn.Dropout(p=sub_mod_i.p))
    return net


def get_cnn_enc_model(data, nfs, layer, **kwargs):
    def f(ni, nf, stride=2): return layer(ni, nf, stride=stride, **kwargs)
    # l1: number of input channels
    l1 = data.train_ds[0][0].shape[0]
    l2 = prev_pow_2(l1*3*3)
    layers = [*f(l1, l2, stride=1),
              *f(l2, l2*2, stride=2),
              *f(l2*2, l2*4, stride=2)]
    nfs = [l2*4] + nfs
    for i in range(len(nfs)-1):
        layers += [*f(nfs[i], nfs[i+1])]
    layers += [nn.AdaptiveAvgPool2d(1), Flatten()]
    return nn.Sequential(*layers)


def get_cnn_dec_model(data, nfs, layer, **kwargs):
    first_upscale = kwargs.get('first_upscale', 2)
    output_fn = kwargs.get('output_fn', None)

    def f(ni, nf, last_layer=False, scale=2):
        if layer.__name__ == 'conv_trans_shuffle':
            return layer(ni, nf, last_layer=last_layer, scale=scale)
        else:
            return layer(ni, nf, last_layer=last_layer, stride=scale)
    # l1: number of output channels
    l1 = data.train_ds[0][0].shape[0]
    l2 = prev_pow_2(l1*3*3)
    nfs = [l2*4] + nfs
    layers = [ResizeBatch(nfs[-1], 1, 1)]
    if layer.__name__ == 'conv_trans_shuffle':
        layers += [*f(nfs[-1], nfs[-2], scale=first_upscale)]
    else:
        layers += [nn.Upsample(scale_factor=first_upscale/2,
                               mode='nearest'), *f(nfs[-1], nfs[-2])]
    for i in reversed(range(len(nfs)-2)):
        layers += [*f(nfs[i+1], nfs[i])]
    layers += [*f(l2*4, l2*2),
               *f(l2*2, l2),
               *f(l2, l1, last_layer=True)]
    if output_fn is not None:
        layers.append(Lambda(output_fn))
    return nn.Sequential(*layers)


class CNNAutoencoderFlexible(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.data = kwargs.get('data')
        self.nfs = kwargs.get('nfs')
        self.layer_enc = kwargs.get('layer_enc')
        self.embd_sz = kwargs.get('embd_sz')
        self.layer_dec = kwargs.get('layer_dec')
        self.tie_weights = kwargs.get('tie_weights', False)
        self.first_upscale = kwargs.get('first_upscale', 2)
        self.output_fn = kwargs.get('output_fn', None)
        self.batch_norm_ae = kwargs.get('batch_norm_ae', True)
        self.dropout_ae = kwargs.get('dropout_ae', 0.0)
        self.encoder = get_cnn_enc_model(self.data, self.nfs, self.layer_enc)
        embedded_enc = lin_bn_drop(
            self.nfs[-1], self.embd_sz, actn=None, bn=self.batch_norm_ae, p=self.dropout_ae)
        embedded_dec = construct_tied_decoder(
            embedded_enc, tie_weights=self.tie_weights)
        self.embedded = AutoencoderFlexible(embedded_enc, embedded_dec)
        self.decoder = get_cnn_dec_model(self.data, self.nfs, self.layer_dec,
                                         first_upscale=self.first_upscale,
                                         output_fn=self.output_fn)

    def encode(self, x):
        encoding_before_embedded = self.encoder(x)
        emb_enc = self.embedded.encoder(encoding_before_embedded)
        return emb_enc

    def decode(self, z):
        emb_rec = self.embedded.reconstruct(z)
        reconstruction = self.decoder(emb_rec)
        return reconstruction

    def forward(self, x):
        emb_enc = self.encode(x)
        reconstruction = self.decode(emb_enc)
        return emb_enc, reconstruction
