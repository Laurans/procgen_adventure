import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rlpyt.models.conv2d import Conv2dModel
from rlpyt.models.dqn.dueling import DuelingHeadModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


@torch.no_grad()
def layer_init(layer, w_scale=np.sqrt(2)):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight.data)
        layer.weight.data.mul_(w_scale)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0)


class DqnModel(torch.nn.Module):
    def __init__(self, image_shape, output_size):
        super(DqnModel, self).__init__()
        c, h, w = image_shape
        fc_sizes = 512
        self.conv = Conv2dModel(
            in_channels=c,
            channels=[32, 64, 64],
            kernel_sizes=[8, 4, 3],
            strides=[4, 2, 1],
            paddings=None,
            use_maxpool=False,
        )

        conv_out_size = self.conv.conv_out_size(h, w)
        self.head = DuelingHeadModel(conv_out_size, fc_sizes, output_size)

        self.apply(layer_init)

    def forward(self, observation, prev_action, prev_reward):
        img = observation.type(torch.float)  # Expect torch.uint8 inputs
        img = img.mul_(1.0 / 255)

        # Infer (presence of) leading dimensions: [T, B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        conv_out = self.conv(img.view(T * B, *img_shape))
        q = self.head(conv_out.view(T * B, -1))

        # Restore leading dimension: [T, B], [B] or [], as input
        q = restore_leading_dims(q, lead_dim, T, B)
        return q


class ImpalaConv(nn.Module):
    def __init__(
        self, *, CHW_shape: tuple, dropout_ratio, use_batch_norm, depths=[16, 32, 32]
    ):
        super(ImpalaConv, self).__init__()
        self.feature_dim = 256

        class ImpalaResidualBlock(nn.Module):
            def __init__(
                self, in_channels, out_channels, dropout_ratio, use_batch_norm
            ):
                super(ImpalaResidualBlock, self).__init__()
                self.conv1 = self.get_conv_layer(
                    in_channels, out_channels, dropout_ratio, use_batch_norm
                )
                self.conv2 = self.get_conv_layer(
                    out_channels, out_channels, dropout_ratio, use_batch_norm
                )

            def get_conv_layer(
                self, in_channels, out_channels, dropout_ratio, use_batch_norm
            ):

                out = [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                ]

                if dropout_ratio > 0:
                    out.append(nn.Dropout2d(p=dropout_ratio))

                if use_batch_norm:
                    out.append(nn.BatchNorm2d(out_channels))

                return nn.ModuleList(out) if len(out) > 1 else out[0]

            def forward(self, x):
                y = F.relu(x)
                y = F.relu(self.conv1(y))
                y = self.conv2(y)
                return x + y

        def conv_sequence(in_channels, out_channels, dropout_ratio, use_batch_norm):
            modulelist = [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                ImpalaResidualBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    dropout_ratio=dropout_ratio,
                    use_batch_norm=use_batch_norm,
                ),
                ImpalaResidualBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    dropout_ratio=dropout_ratio,
                    use_batch_norm=use_batch_norm,
                ),
            ]

            return nn.ModuleList(modulelist)

        layers = []
        in_channels, _, _ = CHW_shape
        for depth in depths:
            layers.append(
                conv_sequence(in_channels, depth, dropout_ratio, use_batch_norm)
            )
            in_channels = depth

        self.conv_body = nn.ModuleList(layers)

        self.out_shape = self._get_conv_out(CHW_shape)

    def _get_conv_out(self, shape):
        o = self.forward(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        y = F.relu(self.conv_body(x))
        return y
