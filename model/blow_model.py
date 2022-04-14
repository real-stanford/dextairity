import numpy as np
import torch
from torch.nn import Sequential as Seq

from .model_utils import MLP, ConvBlock2D


class BlowModel(torch.nn.Module):
    def __init__(self, action_sample_num=64, x_range=0.1, z_rotation=-95, last_action=False):
        super().__init__()
        
        self.action_sample_num = action_sample_num
        self.x_range = x_range
        self.z_rotation = z_rotation
        self.last_action = last_action

        self.image_encoder = Seq(
            ConvBlock2D(3, 32, stride=2), # 038
            ConvBlock2D(32, 64, stride=2), # 64
            ConvBlock2D(64, 128, stride=2), # 32
            ConvBlock2D(128, 256, stride=2), # 16
            ConvBlock2D(256, 256, stride=2), # 8
            ConvBlock2D(256, 256, stride=2), # 4
            torch.nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=0)
        )

        self.action_encoder = MLP([6, 128, 128, 128])
        if self.last_action:
            self.decoder = MLP([256+128+128, 512, 512, 1], last_relu=False)
        else:
            self.decoder = MLP([256+128, 512, 512, 1], last_relu=False)



    def generate_action_samples(self, batch_size):
        num = batch_size * self.action_sample_num
        dx = self.x_range * np.random.rand(num) * np.random.choice([-1, 1], num)
        dy = 0.03 * np.ones(num)
        dz = 0.45 * np.ones(num)
        drx = 30.0 / 180.0 * np.pi * np.random.rand(num) * np.random.choice([-1, 1], num)
        dry = np.zeros(num)
        drz = -105 * np.ones(num) / 180 * np.pi
        action_samples = np.stack([dx, dy, dz, drx, dry, drz], axis=1).reshape([self.action_sample_num, batch_size, 6]).astype(np.float32)
        return action_samples

    def get_forward_actions(self, env_num):
        forward_action = np.array([0, 0.03, 0.45, 0, 0, self.z_rotation / 180 * np.pi])
        return [forward_action] * env_num, [0] * env_num


    def forward(self, scene_input, action=None, last_action=None):
        scene_input = scene_input / 255.0

        batch_size = scene_input.shape[0]
        device = scene_input.device
        image_feature = self.image_encoder(scene_input).squeeze(2).squeeze(2)
        if self.last_action:
            last_action_feature = self.action_encoder(last_action)
        
        if action is not None:
            action_feature = self.action_encoder(action)
            if self.last_action:
                output = self.decoder(torch.cat([image_feature, action_feature, last_action_feature], dim=1)).squeeze(1)
            else:
                output = self.decoder(torch.cat([image_feature, action_feature], dim=1)).squeeze(1)
            return output
        else:
            action_samples = self.generate_action_samples(batch_size)
            predictions = list()
            for i in range(self.action_sample_num):
                action_feature = self.action_encoder(torch.from_numpy(action_samples[i]).to(device))
                if self.last_action:
                    prediction = self.decoder(torch.cat([image_feature, action_feature, last_action_feature], dim=1)).squeeze(1)
                else:
                    prediction = self.decoder(torch.cat([image_feature, action_feature], dim=1)).squeeze(1)
                predictions.append(prediction)
            return action_samples, torch.stack(predictions)