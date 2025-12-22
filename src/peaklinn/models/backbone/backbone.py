from typing import Union as U
from collections import defaultdict, OrderedDict

import torch.nn as nn

from .selective_sequential import SelectiveSequential
from .cbam_1d import CBAM1D


def add_layer(layer, layers, cnt):
    name = type(layer).__name__
    cnt[name] += 1
    lname = f'{name}_{cnt[name]}'
    layers[lname] = layer
    return lname


def conv_module(cfg,
                in_channels,
                encoder=True,
                downsampling='str',
                kernel_size: U[int, list] = 3,
                stride: U[int, list] = 1,
                activation=nn.LeakyReLU,
                cbam_kernel_size=7,
                selective=False):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * len(cfg)
    if isinstance(stride, int):
        stride = [stride] * len(cfg)

    conv_class = nn.Conv1d if encoder else nn.ConvTranspose1d

    layers = OrderedDict()
    cnt = defaultdict(int)
    conv_num = 0
    to_select = []
    for v in cfg:
        if v == 'D':
            if downsampling == 'max':
                add_layer(nn.MaxPool1d(kernel_size=2, stride=2), layers, cnt)
            elif downsampling == 'avg':
                add_layer(nn.AvgPool1d(kernel_size=2, stride=2), layers, cnt)
            elif downsampling == 'str':
                name = conv_class.__name__
                last_conv = f'{name}_{cnt[name]}'
                layers[last_conv].stride = (layers[last_conv].stride[0] * 2,)
        elif v == 'U':
            name = conv_class.__name__
            last_conv = f'{name}_{cnt[name]}'
            layers[last_conv].stride = (layers[last_conv].stride[0] * 2,)
            layers[last_conv].output_padding = (layers[last_conv].stride[0] - 1,)
        elif v == 'B':
            add_layer(nn.BatchNorm1d(in_channels), layers, cnt)
        elif v == 'A':
            add_layer(activation(inplace=True), layers, cnt)
        elif v == 'CBAM':
            add_layer(CBAM1D(in_channels, 16, cbam_kernel_size), layers, cnt)
        else:
            ks = kernel_size[conv_num]
            conv1d = conv_class(in_channels, v, kernel_size=ks, padding=ks // 2,
                                stride=stride[conv_num])
            to_select.append(add_layer(conv1d, layers, cnt))
            in_channels = v
            conv_num += 1
    # add last layer to selection
    to_select.append(next(reversed(layers.keys())))
    if selective:
        return SelectiveSequential(to_select, layers), in_channels
    else:
        return nn.Sequential(*[layers[l] for l in layers]), in_channels
