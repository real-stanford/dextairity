import numpy as np
import torch
from torch.nn import Sequential as Seq

from .model_utils import MLP, ConvBlock2D


class BagModel(torch.nn.Module):
    def __init__(self, input_dim=3, image_dim=1):
        super().__init__()
        self.image_dim = image_dim

        self.image_encoder = Seq(
            ConvBlock2D(image_dim, 8, stride=2), # 256
            ConvBlock2D(8, 16, stride=2), # 128
            ConvBlock2D(16, 32, stride=2), # 64
            ConvBlock2D(32, 64, stride=2), # 32
            ConvBlock2D(64, 128, stride=2), # 16
            ConvBlock2D(128, 128, stride=2), # 8
            ConvBlock2D(128, 128, stride=2), # 4
            torch.nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=0)
        )
        self.action_encoder = MLP([input_dim, 128, 128, 128])
        self.decoder = MLP([128+128, 256, 256, 1], last_relu=False)
            

    def forward(self, scene_input, action):
        image_feature = self.image_encoder(scene_input).squeeze(2).squeeze(2)
        if isinstance(action, list):
            output = [self.decoder(torch.cat([image_feature, self.action_encoder(x)], dim=1)).squeeze(1) for x in action]
        else:
            output = self.decoder(torch.cat([image_feature, self.action_encoder(action)], dim=1)).squeeze(1)
        
        return output
