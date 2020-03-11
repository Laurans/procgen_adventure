import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from procgen_adventure.network.utils import layer_init

mapping = {}


def register(decorated):
    mapping[decorated.__name__] = decorated

    return decorated


@register
class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x


@register
class NatureConv(nn.Module):
    def __init__(self, *, CHW_shape: tuple, **kwargs):
        super(NatureConv, self).__init__()
        self.feature_dim = 512

        in_channels, _, _ = CHW_shape
        self.conv1 = layer_init(
            nn.Conv2d(
                in_channels=in_channels, out_channels=32, kernel_size=8, stride=4
            ),
            w_scale=np.sqrt(2),
        )
        self.conv2 = layer_init(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            w_scale=np.sqrt(2),
        )
        self.conv3 = layer_init(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            w_scale=np.sqrt(2),
        )

        out_shape = self._get_conv_out(CHW_shape)
        self.fc4 = layer_init(
            nn.Linear(in_features=out_shape, out_features=self.feature_dim),
            w_scale=np.sqrt(2),
        )

    def _get_conv_out(self, shape):
        o = self._conv_forward(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def _conv_forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        return out

    def forward(self, x):
        y = self._conv_forward(x).view(x.size(0), -1)
        y = F.relu(self.fc4(y))
        return y


@register
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

        out_shape = self._get_conv_out(CHW_shape)
        self.fc = nn.Linear(out_shape, self.feature_dim)

    def _conv_forward(self, x):
        return F.relu(self.conv_body(x))

    def _get_conv_out(self, shape):
        o = self._conv_forward(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        y = self._conv_forward(x).view(x.view(0), -1)
        y = F.relu(self.fc(y))
        return y


def body_factory(name):
    if name in mapping:
        return mapping[name]
    else:
        raise ValueError(f"Unknown network type: {name}")
