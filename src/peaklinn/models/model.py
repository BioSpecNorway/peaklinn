import torch
import torch.nn as nn

from peaklinn.configs.model_config import MieDSAEConfig
from peaklinn.models.backbone.backbone import conv_module


class MieDescatteringAutoencoder(nn.Module):
    def __init__(self, config: MieDSAEConfig):
        super().__init__()

        self.cfg = config
        self.in_channels = 1
        if self.cfg.with_wavelengths:
            self.in_channels += 1

        self.enc_cfg = self.get_enc_dec_config(encoder=True)
        self.encoder, self.encoder_end_channels = conv_module(
            cfg=self.enc_cfg,
            downsampling=self.cfg.downsampling,
            in_channels=self.in_channels,
            kernel_size=self.cfg.kernel_size,
            cbam_kernel_size=self.cfg.cbam_kernel_size,
            selective=self.cfg.skip_connections
        )

        self.dec_cfg = self.get_enc_dec_config(encoder=False)
        self.decoder, self.decoder_end_channels = conv_module(
            self.dec_cfg,
            in_channels=self.encoder_end_channels,
            encoder=False,
            kernel_size=self.cfg.kernel_size,
            cbam_kernel_size=self.cfg.cbam_kernel_size,
            selective=self.cfg.skip_connections,
        )


        self.spectra_output = self.create_convolutional_head()

    def smart_forward(self, measured, wns):
        if self.cfg.float_precision:
            measured, wns = measured.float(), wns.float()
        else:
            measured, wns = measured.double(), wns.double()

        x = measured
        if self.cfg.zero_mean_qext:
            x = x - x.mean(dim=-1, keepdim=True)
        if self.cfg.one_scale_qext:
            x = x - x.min(dim=-1, keepdim=True)[0]
            x = x / (x.max(dim=-1, keepdim=True)[0] + 1e-10)
        if self.cfg.with_wavelengths:
            x = torch.cat((x, 10e3/wns), dim=1)

        return self.forward(x)

    def forward(self, x):
        x = self.encoder(x)

        if self.cfg.skip_connections:
            x_sel, last = x[:-1], x[-1]
            adds = reversed(x_sel[1::2])
            to_layers_out = self.decoder.to_select[:-1][::2]
            add_to_output = {lname: out
                             for lname, out in zip(to_layers_out, adds)}
            x = self.decoder(last, add_to_output)[-1]
        else:
            x_sel, last = [], x
            x = self.decoder(last)

        pure = self.spectra_output(x)

        return pure

    def get_enc_dec_config(self, encoder=True):
        assert self.cfg.n_blocks > 0

        n_flts = self.cfg.start_n_filters if encoder else self.encoder_end_channels
        cfg = []
        for _ in range(self.cfg.n_blocks):
            block = [n_flts]
            if self.cfg.batch_norm:
                block.append('B')
            block.append('A')
            if self.cfg.cbam:
                block.append('CBAM')

            if encoder:
                block = block * 2 + ['D']
                n_flts *= 2
            else:
                block = block * 2 + ['U']
                n_flts //= 2

            cfg.extend(block)

        return cfg

    def create_convolutional_head(self):
        in_channels = self.cfg.start_n_filters
        end_layers = []

        end_layers.append(
            nn.Conv1d(self.decoder_end_channels,
                      in_channels,
                      self.cfg.kernel_size,
                      padding=self.cfg.kernel_size // 2)
        )

        for _ in range(self.cfg.n_end_convs - 2):
            end_layers.append(nn.Conv1d(in_channels, in_channels,
                                        self.cfg.kernel_size,
                                        padding=self.cfg.kernel_size // 2))
        end_layers.append(nn.Conv1d(in_channels, 1,
                                    self.cfg.kernel_size,
                                    padding=self.cfg.kernel_size // 2))
        return nn.Sequential(*end_layers)
