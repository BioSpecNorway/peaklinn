import torch.nn as nn
import torch


class CBAM1D(nn.Module):

    def __init__(self, n_channels_in, reduction_ratio, kernel_size):
        super(CBAM1D, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        self.channel_attention = ChannelAttention(n_channels_in,
                                                  reduction_ratio)
        self.spatial_attention = SpatialAttention1D(kernel_size)

    def forward(self, f):
        chan_att = self.channel_attention(f)
        fp = chan_att * f
        spat_att = self.spatial_attention(fp)
        fpp = spat_att * fp
        return fpp


class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention1D, self).__init__()
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, "Odd kernel size required"
        self.conv = nn.Conv1d(in_channels=2, out_channels=1,
                              kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        pool = torch.cat([max_pool, avg_pool], dim=1)
        conv = self.conv(pool)

        conv = conv.repeat(1, x.size()[1], 1)
        att = torch.sigmoid(conv)
        return att


class ChannelAttention(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = int(self.n_channels_in / self.reduction_ratio)
        if self.middle_layer_size < 2:
            self.middle_layer_size = 2

        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.middle_layer_size),
            nn.ReLU(),
            nn.Linear(self.middle_layer_size, self.n_channels_in)
        )

    def forward(self, x):
        inp_shape = x.size()
        max_pool, _ = torch.max(x, dim=-1)
        avg_pool = torch.mean(x, dim=-1)

        max_pool_bck = self.bottleneck(max_pool)
        avg_pool_bck = self.bottleneck(avg_pool)

        pool_sum = avg_pool_bck + max_pool_bck

        sig_pool = torch.sigmoid(pool_sum)
        sig_pool = sig_pool.unsqueeze(2)

        out = sig_pool.repeat(1, 1, inp_shape[2])
        return out
