import torch
import numpy as np
import sys
from torch import nn
import torch.nn.functional as F

# Model is based on:
# U-Net: Convolutional Networks for Biomedical Image Segmentation; https://arxiv.org/abs/1505.04597
# Squeeze U-Net: A Memory and Energy Efficient Image Segmentation Network; DOI: 10.1109/CVPRW50498.2020.00190
# Noise2Stack: Improving Image Restoration by Learning from Volumetric Data; https://arxiv.org/abs/2011.05105

# Multi-resolution convolutional neural networks for inverse problems; https://doi.org/10.1038/s41598-020-62484-z


class SqueezeUNet(nn.Module):
    def __init__(self, depth=5, wf=6, dropout=0.5, NN=1):
        super(SqueezeUNet, self).__init__()
        self.depth = depth
        self.nf = 2**wf
        self.dropout = dropout
        self.NN = NN
        self.effective_depth = self.depth

        # Encoder
        self.down_path = nn.ModuleList()

        # First layer
        ini_block = []
        if self.NN > 7:
            print("Number of neighbors unsupported")
            sys.exit(-1)
        if self.NN % 2 == 0:
            # unsupervised
            if self.NN == 2:
                ini_block.append(
                    nn.Conv3d(1, self.nf, kernel_size=(2, 3, 3), padding=(0, 1, 1))
                )
                nn.init.kaiming_normal_(ini_block[-1].weight, nonlinearity="leaky_relu")
                ini_block.append(nn.LeakyReLU(inplace=True))
                ini_block.append(nn.BatchNorm3d(self.nf))
            elif self.NN == 4:
                ini_block.append(
                    nn.Conv3d(1, self.nf, kernel_size=(3, 3, 3), padding=(0, 1, 1))
                )
                nn.init.kaiming_normal_(ini_block[-1].weight, nonlinearity="leaky_relu")
                ini_block.append(nn.LeakyReLU(inplace=True))
                ini_block.append(
                    nn.Conv3d(
                        self.nf, self.nf, kernel_size=(2, 3, 3), padding=(0, 1, 1)
                    )
                )
                nn.init.kaiming_normal_(ini_block[-1].weight, nonlinearity="leaky_relu")
                ini_block.append(nn.LeakyReLU(inplace=True))
                ini_block.append(nn.BatchNorm3d(self.nf))
        else:
            # supervised
            if self.NN == 1:
                ini_block.append(nn.Conv2d(1, self.nf, kernel_size=3, padding=1))
                nn.init.kaiming_normal_(ini_block[-1].weight, nonlinearity="leaky_relu")
                ini_block.append(nn.LeakyReLU(inplace=True))
                ini_block.append(nn.BatchNorm2d(self.nf))
            elif self.NN == 3:
                ini_block.append(
                    nn.Conv3d(1, self.nf, kernel_size=(3, 3, 3), padding=(0, 1, 1))
                )
                nn.init.kaiming_normal_(ini_block[-1].weight, nonlinearity="leaky_relu")
                ini_block.append(nn.LeakyReLU(inplace=True))
                ini_block.append(nn.BatchNorm3d(self.nf))
            elif self.NN == 5:
                ini_block.append(
                    nn.Conv3d(1, self.nf, kernel_size=(3, 3, 3), padding=(0, 1, 1))
                )
                nn.init.kaiming_normal_(ini_block[-1].weight, nonlinearity="leaky_relu")
                ini_block.append(nn.LeakyReLU(inplace=True))
                ini_block.append(
                    nn.Conv3d(
                        self.nf, self.nf, kernel_size=(3, 3, 3), padding=(0, 1, 1)
                    )
                )
                nn.init.kaiming_normal_(ini_block[-1].weight, nonlinearity="leaky_relu")
                ini_block.append(nn.LeakyReLU(inplace=True))
                ini_block.append(nn.BatchNorm3d(self.nf))
            elif self.NN == 7:
                ini_block.append(
                    nn.Conv3d(1, self.nf, kernel_size=(3, 3, 3), padding=(0, 1, 1))
                )
                nn.init.kaiming_normal_(ini_block[-1].weight, nonlinearity="leaky_relu")
                ini_block.append(nn.LeakyReLU(inplace=True))
                ini_block.append(
                    nn.Conv3d(
                        self.nf, self.nf, kernel_size=(3, 3, 3), padding=(0, 1, 1)
                    )
                )
                nn.init.kaiming_normal_(ini_block[-1].weight, nonlinearity="leaky_relu")
                ini_block.append(nn.LeakyReLU(inplace=True))
                ini_block.append(
                    nn.Conv3d(
                        self.nf, self.nf, kernel_size=(3, 3, 3), padding=(0, 1, 1)
                    )
                )
                nn.init.kaiming_normal_(ini_block[-1].weight, nonlinearity="leaky_relu")
                ini_block.append(nn.LeakyReLU(inplace=True))
                ini_block.append(nn.BatchNorm3d(self.nf))

        ini_block = nn.Sequential(*ini_block)
        self.down_path.append(ini_block)

        # other encoder layers
        prev_channels = self.nf
        for i in range(1, depth):
            block = FireBlock(prev_channels, 2 ** (wf + i), dropout=self.dropout)
            self.down_path.append(block)
            prev_channels = 2 ** (wf + i)

        # Decoder
        self.up_path = nn.ModuleList()
        self.up_path_conv = nn.ModuleList()
        for i in reversed(range(self.depth - 1)):
            block = FireUpBlock(prev_channels, 2 ** (wf + i), dropout=self.dropout)
            self.up_path.append(block)
            prev_channels = 2 ** (wf + i)
            self.up_path_conv.append(nn.Conv2d(prev_channels, 1, kernel_size=1))

        # freezes unused parameters
        self.set_effective_depth(self.effective_depth)

        # last layer
        # self.last = nn.Conv2d(prev_channels, 1, kernel_size=1)

    def forward(self, x):
        B, D, W, H = x.shape
        if self.NN > 1:
            y = torch.reshape(x, (B, 1, D, W, H))
        else:
            y = torch.clone(x)

        # encode
        skip_blocks = []
        for i, down in enumerate(self.down_path):
            y = down(y)
            if i == 0 and self.NN > 1:
                y = torch.reshape(y, (B, self.nf, W, H))
            if i != len(self.down_path) - 1:
                skip_blocks.append(y)
                y = F.max_pool2d(y, 2)

        # decode
        for i, up in enumerate(self.up_path):
            y = up(y, skip_blocks[-i - 1])
            if i + 2 == self.effective_depth:
                if self.effective_depth == self.depth:
                    y = self.up_path_conv[i](y)
                else:
                    y = self.up_path_conv[i](y)
                    y = F.interpolate(
                        y, size=(self.nf, self.nf), mode="bicubic", align_corners=True
                    )
                break

        return y

    def set_effective_depth(self, d):
        self.effective_depth = min(d, self.depth)
        # This is erroneous
        # for i in range(len(self.up_path_conv)):
        #     for p in self.up_path_conv[i].parameters():
        #         if i+2 == self.effective_depth:
        #             p.requires_grad = True
        #         else:
        #             p.requires_grad = False
        # for i in range(len(self.up_path)):
        #     for p in self.up_path[i].parameters():
        #         if i+2 <= self.effective_depth:
        #             p.requires_grad = True
        #         else:
        #             p.requires_grad = False
        print("Effective depth updated to: ", self.effective_depth)

    def update_dropout(self, p):
        self.dropout = max(0.01, min(p, 0.5))
        for module in self.up_path:
            module.conv_block.dropout = self.dropout
        for module in self.down_path[1:]:
            module.dropout = self.dropout
        print("updated dropout to: ", self.dropout)

    def write_model(self, filename):
        # should include learning & epoch read as part of model state!
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "NN": self.NN,
                "nf": self.nf,
                "dropout": self.dropout,
                "depth": self.depth,
                "effective_depth": self.effective_depth,
            },
            filename,
        )

    def load_model(self, filename):
        ckpt = torch.load(filename)
        NN = ckpt["NN"]
        depth = ckpt["depth"]
        nf = ckpt["nf"]
        dropout = ckpt["dropout"]
        effective_depth = ckpt["effective_depth"]
        self = SqueezeUNet(depth=depth, wf=int(np.log2(nf)), dropout=dropout, NN=NN)
        model_state_dict = ckpt["model_state_dict"]
        self.load_state_dict(model_state_dict)
        self.set_effective_depth(effective_depth)
        print(f"\nLoaded model: ", filename)
        return self


class FireBlock(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.5):
        super(FireBlock, self).__init__()
        self.dropout = dropout
        squeeze_size = in_size // 2
        expand_size = out_size // 2

        layer0 = []
        layer0.append(nn.Conv2d(in_size, squeeze_size, kernel_size=1))
        nn.init.kaiming_normal_(
            layer0[0].weight, mode="fan_in", nonlinearity="leaky_relu"
        )
        layer0.append(nn.LeakyReLU(inplace=True))
        layer0.append(nn.BatchNorm2d(squeeze_size))
        self.layer0 = nn.Sequential(*layer0)

        layer0a = []
        layer0a.append(nn.Conv2d(squeeze_size, expand_size, kernel_size=1))
        nn.init.kaiming_normal_(
            layer0a[0].weight, mode="fan_in", nonlinearity="leaky_relu"
        )
        layer0a.append(nn.LeakyReLU(inplace=True))
        self.layer0a = nn.Sequential(*layer0a)

        layer0b = []
        layer0b.append(nn.Conv2d(squeeze_size, expand_size, kernel_size=3, padding=1))
        nn.init.kaiming_normal_(
            layer0b[0].weight, mode="fan_in", nonlinearity="leaky_relu"
        )
        layer0b.append(nn.LeakyReLU(inplace=True))
        self.layer0b = nn.Sequential(*layer0b)

        layer1 = []
        layer1.append(nn.Conv2d(out_size, squeeze_size, kernel_size=1))
        nn.init.kaiming_normal_(
            layer1[0].weight, mode="fan_in", nonlinearity="leaky_relu"
        )
        layer1.append(nn.LeakyReLU(inplace=True))
        layer1.append(nn.BatchNorm2d(squeeze_size))
        self.layer1 = nn.Sequential(*layer1)

        layer1a = []
        layer1a.append(nn.Conv2d(squeeze_size, expand_size, kernel_size=1))
        nn.init.kaiming_normal_(
            layer1a[0].weight, mode="fan_in", nonlinearity="leaky_relu"
        )
        layer1a.append(nn.LeakyReLU(inplace=True))
        self.layer1a = nn.Sequential(*layer1a)

        layer1b = []
        layer1b.append(nn.Conv2d(squeeze_size, expand_size, kernel_size=3, padding=1))
        nn.init.kaiming_normal_(
            layer1b[0].weight, mode="fan_in", nonlinearity="leaky_relu"
        )
        layer1b.append(nn.LeakyReLU(inplace=True))
        self.layer1b = nn.Sequential(*layer1b)

    def forward(self, x):
        y = self.layer0(x)
        y = torch.cat([self.layer0a(y), self.layer0b(y)], 1)
        y = self.layer1(y)
        y = torch.cat([self.layer1a(y), self.layer1b(y)], 1)
        return F.dropout2d(y, p=self.dropout, training=self.training)


class FireUpBlock(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.5):
        super(FireUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        self.conv_block = FireBlock(in_size, out_size, dropout=dropout)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        return self.conv_block(torch.cat([up, crop1], 1))
