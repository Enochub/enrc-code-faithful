import torch
import torch.nn.functional as F
import numpy as np


class TiedLinear(torch.nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
        Adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html
        for weight tying. 
        The use of this module is required otherwise there are problems with updates of the
        bias parameters.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        tied_weights: weights that should be reused
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            `(out_features x in_features)`
        bias:   the learnable bias of the module of shape `(out_features)`
    """

    def __init__(self, in_features, out_features, tied_weight=None, bias=True):
        super(TiedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if tied_weight is not None:
            if tied_weight.shape[1] != self.in_features or \
                    tied_weight.shape[0] != self.out_features:
                raise ValueError(
                    "Shape of tied weights does not correspond to in_features and out_features")
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.out_features)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, tied_weight):
        """
        Need to forward tied_weight every time. When using
        torch.nn.Parameter and tie weights manually they are update,
        which is not preferred for weight tying as the gradients should
        be accumulated and only one update should be done.
        """
        return F.linear(input, tied_weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class AutoencoderFlexible(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoencoderFlexible, self).__init__()
        self.encoder = torch.nn.Sequential(*encoder)
        self.decoder = torch.nn.Sequential(*decoder)

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
        return reconstruction
