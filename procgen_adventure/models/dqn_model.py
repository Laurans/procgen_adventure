import torch
from rlpyt.models.conv2d import Conv2dModel
from rlpyt.models.dqn.dueling import DuelingHeadModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


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
            paddings=[0, 1, 1],
            use_maxpool=False,
        )

        conv_out_size = self.conv.conv_out_size(h, w)
        self.head = DuelingHeadModel(conv_out_size, fc_sizes, output_size)

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
