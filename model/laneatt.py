import numpy as np
import os 
import torch
import yaml

import torch.nn as nn
from torchvision import models

class LaneATT():
    def __init__(self, backbone='resnet18', anchor_feat_channels=64, S=72, img_size=(640, 360)) -> None:
        self.config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), 'laneatt_config.yaml'))) # Load config file
        self.backbone = backbone # Set backbone
        self.anchor_feat_channels = anchor_feat_channels # Set anchor feature channels
        self.n_offsets = S # Number of offsets

        # Image size
        self.img_w, self.img_h = img_size

        # Anchor angles, same ones used in Line-CNN
        self.left_angles = [72., 60., 49., 39., 30., 22.]
        self.right_angles = [108., 120., 131., 141., 150., 158.]
        self.bottom_angles = [165., 150., 141., 131., 120., 108., 100., 90., 80., 72., 60., 49., 39., 30., 15.]

        self.anchor_ys = torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32)
        self.anchor_cut_ys = torch.linspace(1, 0, steps=self.fmap_h, dtype=torch.float32)

        self.anchors, self.anchors_cut = self.generate_anchors(lateral_n=72, bottom_n=128)
        print(self.anchors.shape, self.anchors_cut.shape)

    @property
    def backbone(self):
        return self._backbone # Backbone Getter
    
    @backbone.setter
    def backbone(self, value):
        value = value.lower()
        # Check if value is in the list of backbones in config file
        if value not in self.config['backbones']:
            raise ValueError(f'Backbone must be one of {self.config['backbones']}')
        # Set pretrained backbone according to pytorch requirements without the average pooling and fully connected layer
        self._backbone = torch.nn.Sequential(*list(models.__dict__[value](weights=f'{value.replace('resnet', 'ResNet')}_Weights.DEFAULT').children())[:-2])
        self.fmap_h = self._backbone(torch.randn(1, 3, 224, 224)).shape[2] # Feature Map Height


    def forward(self, x):
        feature_map = self.backbone(x) # ResNet backbone Feature Volume
        pooled_map = nn.Conv2d(feature_map.shape[1], self.anchor_feat_channels, kernel_size=1)(feature_map) # Dimensionality Reduction Feature Volume
        #anchors, anchors_cut = self.cut_anchor_features(pooled_map)
        return x
    
    def cut_anchor_features(self, feature_volume):
        # batch_size = feature_volume.shape[0]
        # n_proposals = len(self.anchors)
        pass

    def generate_anchors(self, lateral_n, bottom_n):
        left_anchors, left_cut = self.generate_side_anchors(self.left_angles, x=0., nb_origins=lateral_n)
        right_anchors, right_cut = self.generate_side_anchors(self.right_angles, x=1., nb_origins=lateral_n)
        bottom_anchors, bottom_cut = self.generate_side_anchors(self.bottom_angles, y=1., nb_origins=bottom_n)

        return torch.cat([left_anchors, bottom_anchors, right_anchors]), torch.cat([left_cut, bottom_cut, right_cut])
    
    def generate_side_anchors(self, angles, nb_origins, x=None, y=None):
        if x is None and y is not None:
            starts = [(x, y) for x in np.linspace(1., 0., num=nb_origins)]
        elif x is not None and y is None:
            starts = [(x, y) for y in np.linspace(1., 0., num=nb_origins)]
        else:
            raise Exception('Please define exactly one of `x` or `y` (not neither nor both)')

        n_anchors = nb_origins * len(angles)

        # each row, first for x and second for y:
        # 2 scores, 1 start_y, start_x, 1 lenght, S coordinates, score[0] = negative prob, score[1] = positive prob
        anchors = torch.zeros((n_anchors, 2 + 2 + 1 + self.n_offsets))
        anchors_cut = torch.zeros((n_anchors, 2 + 2 + 1 + self.fmap_h))
        for i, start in enumerate(starts):
            for j, angle in enumerate(angles):
                k = i * len(angles) + j
                anchors[k] = self.generate_anchor(start, angle)
                anchors_cut[k] = self.generate_anchor(start, angle, cut=True)

        return anchors, anchors_cut
    
    def generate_anchor(self, start, angle, cut=False):
        if cut:
            anchor_ys = self.anchor_cut_ys
            anchor = torch.zeros(2 + 2 + 1 + self.fmap_h)
        else:
            anchor_ys = self.anchor_ys
            anchor = torch.zeros(2 + 2 + 1 + self.n_offsets)
        angle = angle * np.pi / 180.  # degrees to radians
        start_x, start_y = start
        anchor[2] = 1 - start_y
        anchor[3] = start_x
        anchor[5:] = (start_x + (1 - anchor_ys - 1 + start_y) / np.tan(angle)) * self.img_w

        return anchor
    
laneatt = LaneATT('resnet101')
laneatt.forward(torch.randn(1, 3, 640, 320))