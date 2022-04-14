import numpy as np
import torch
from torch import nn
from torch.nn import Sequential as Seq

from .deeplab import DeepLabv3_plus
from .model_utils import ResBlock2D, rotate_tensor2d, ConvBlock2D


class GraspModel(torch.nn.Module):
    def __init__(self, model_type, rotation_num=16):
        super().__init__()

        self.model_type = model_type
        self.rotation_num = rotation_num

        if self.model_type == 'heuristic':
            self.network = Seq(
                DeepLabv3_plus(nInputChannels=3, n_classes=32, os=8, pretrained=False, _print=False),
                nn.ReLU(),
                nn.Conv2d(32, 1, 1)
            )
        elif self.model_type in ['flingbot', 'pick_and_place']:
            self.network = Seq(
                ConvBlock2D(3, 16, norm=True, relu=True),
                ResBlock2D(16, 16),
                ResBlock2D(16, 16),
                ResBlock2D(16, 16),
                ResBlock2D(16, 16),
                ResBlock2D(16, 16),
                ResBlock2D(16, 16),
                ResBlock2D(16, 16),
                ResBlock2D(16, 16),
                ResBlock2D(16, 16),
                nn.Conv2d(16, 1, 1)
            )


    def forward(self, scene_input, rotation_angle=None):
        scene_input = scene_input / 255.0
        
        # If rotation angle (in radians) is not specified, do forward pass with all rotations
        if rotation_angle is None:
            rotation_angle = np.arange(self.rotation_num)*(2 * np.pi / self.rotation_num) if self.model_type == 'pick_and_place' else np.arange(self.rotation_num)*(np.pi / self.rotation_num)
            rotation_angle = np.array([[x] * scene_input.size(0) for x in rotation_angle])
        
        rotated_scenes = [rotate_tensor2d(scene_input, -angles, padding_mode='border') for angles in rotation_angle] # rotate scene tensor clockwise <=> rotate scene counter-clockwise
        outputs = [self.network(scene) for scene in rotated_scenes]
        outputs_rotated_back = [rotate_tensor2d(scene, angles, padding_mode='border') for scene, angles in zip(outputs, rotation_angle)] # rotate scene tensor counter-clockwise <=> rotate scene clockwise
        
        output = torch.cat(outputs_rotated_back, dim=1)
        
        return output