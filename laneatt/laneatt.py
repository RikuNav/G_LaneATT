import os 
import torch
import torch.nn as nn
import yaml
import random
import numpy as np

import utils
from torchvision import models
from tqdm import tqdm, trange

class LaneATT(nn.Module):
    def __init__(self, config_file=os.path.join(os.path.dirname(__file__), 'config', 'laneatt.yaml')) -> None:
        # Call parent constructor
        super(LaneATT, self).__init__()
        # Load LaneATT config file
        self.__laneatt_config_file = config_file
        self.__laneatt_config = yaml.safe_load(open(config_file))
        # Load backbones config file
        self.__backbones_config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), 'config', 'backbones.yaml')))
        # Set anchor feature channels
        self.__feature_volume_channels = self.__laneatt_config['feature_volume_channels']
        # Set anchor y steps
        self.__anchor_y_steps = self.__laneatt_config['anchor_steps']['y']
        # Set anchor x steps
        self.__anchor_x_steps = self.__laneatt_config['anchor_steps']['x']
        # Set image width
        self.__img_w = self.__laneatt_config['image_size']['width']
        # Set image height
        self.__img_h = self.__laneatt_config['image_size']['height']
        # Create anchor feature dimensions variables but they will be defined after the backbone is created
        self.__feature_volume_height = None
        self.__feature_volume_width = None

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Creates the backbone and moves it to the device
        self.backbone = self.__laneatt_config['backbone']

        # Generate Anchors Proposals
        self.__anchors_image, self.__anchors_feature_volume = utils.generate_anchors(lateral_n=self.__anchor_y_steps, 
                                                                                    bottom_n=self.__anchor_x_steps,
                                                                                    left_angles=self.__laneatt_config['anchor_angles']['left'],
                                                                                    right_angles=self.__laneatt_config['anchor_angles']['right'],
                                                                                    bottom_angles=self.__laneatt_config['anchor_angles']['bottom'],
                                                                                    y_steps=self.__anchor_y_steps,
                                                                                    feature_volume_height=self.__feature_volume_height)

        # Move the anchors to the device
        self.__anchors_image = self.__anchors_image.to(self.device)
        self.__anchors_feature_volume = self.__anchors_feature_volume.to(self.device)
        
        # Pre-Compute Indices for the Anchor Pooling
        self.__anchors_z_cut_indices, self.__anchors_y_cut_indices, self.__anchors_x_cut_indices, self.__invalid_mask = utils.compute_anchor_cut_indices(self.__anchors_feature_volume,
                                                                                                                                                        self.__feature_volume_channels, 
                                                                                                                                                        self.__feature_volume_height, 
                                                                                                                                                        self.__feature_volume_width)

        # Move the indices to the device
        self.__anchors_z_cut_indices = self.__anchors_z_cut_indices.to(self.device)
        self.__anchors_y_cut_indices = self.__anchors_y_cut_indices.to(self.device)
        self.__anchors_x_cut_indices = self.__anchors_x_cut_indices.to(self.device)

        # Fully connected layer of the attention mechanism that takes a single anchor proposal for all the feature maps as input and outputs a score 
        # for each anchor proposal except itself. The score is computed using a softmax function.
        self.__attention_layer = nn.Sequential(nn.Linear(self.__feature_volume_channels * self.__feature_volume_height, len(self.__anchors_feature_volume) - 1),
                                                nn.Softmax(dim=1)).to(self.device)
        
        # Convolutional layer for the classification task
        self.__cls_layer = nn.Linear(2 * self.__feature_volume_channels * self.__feature_volume_height, 2).to(self.device)
        self.__reg_layer = nn.Linear(2 * self.__feature_volume_channels * self.__feature_volume_height, self.__anchor_y_steps + 1).to(self.device)

    @property
    def backbone(self):
        """
            Getter for backbone

            Args:
                None

            Returns:
                nn.Sequential: Pretrained backbone
        """
        return self.__backbone
    
    @backbone.setter
    def backbone(self, value):
        """
            Setter for backbone

            Args:
                value (str): Backbone name

            Returns:
                None
        """
        # Lower the value to avoid case sensitivity
        value = value.lower()

        # Check if value is in the list of backbones in config file
        if value not in self.__backbones_config['backbones']:
            raise ValueError(f'Backbone must be one of {self.config["backbones"]}')
        
        # Set pretrained backbone according to pytorch requirements without the average pooling and fully connected layer
        self.__backbone = nn.Sequential(*list(models.__dict__[value](weights=f'{value.replace("resnet", "ResNet")}_Weights.DEFAULT').children())[:-2],)

        # Runs backbone (on cpu) once to get output data 
        backbone_dimensions = self.__backbone(torch.randn(1, 3, self.__img_h, self.__img_w)).shape

        # Extracts feature volume height and width
        self.__feature_volume_height = backbone_dimensions[2]
        self.__feature_volume_width = backbone_dimensions[3]

        # Join the backbone and the convolutional layer for dimensionality reduction
        self.__backbone = nn.Sequential(self.__backbone, nn.Conv2d(backbone_dimensions[1], self.__feature_volume_channels, kernel_size=1))

        # Move the model to the device
        self.__backbone.to(self.device)
    
    @property
    def config(self):
        """
            Getter for config

            Args:
                None

            Returns:
                dict: LaneATT config
        """
        return self.__laneatt_config_file

    def forward(self, x):
        """
            Forward pass of the model

            Args:
                x (torch.Tensor): Input image
        """
        # Move the input to the device
        x = x.to(self.device)
        # Gets the feature volume from the backbone with a dimensionality reduction layer
        feature_volumes = self.backbone(x)
        # Cuts the anchor features from the feature volumes
        batch_anchor_features = self.__cut_anchor_features(feature_volumes)
        # Join proposals from all images into a single proposals features batch
        batch_anchor_features = batch_anchor_features.view(-1, self.__feature_volume_channels * self.__feature_volume_height)

        # Compute attention scores and reshape them to the original batch size
        attention_scores = self.__attention_layer(batch_anchor_features).reshape(x.shape[0], len(self.__anchors_feature_volume), -1)
        # Generate the attention matrix to be used to store the attention scores
        attention_matrix = torch.eye(attention_scores.shape[1], device=x.device).repeat(x.shape[0], 1, 1)
        # Gets the indices of the non diagonal elements of the attention matrix
        non_diag_indices = torch.nonzero(attention_matrix == 0., as_tuple=False)
        # Makes the entire attention matrix to be zero
        attention_matrix[:] = 0
        # Assigns the attention scores to the attention matrix ignoring the self attention scores as they are not calculated
        # This way we can have a matrix with the attention scores for each anchor proposal
        attention_matrix[non_diag_indices[:, 0], non_diag_indices[:, 1], non_diag_indices[:, 2]] = attention_scores.flatten()

        # Reshape the batch anchor features to the original batch size
        batch_anchor_features = batch_anchor_features.reshape(x.shape[0], len(self.__anchors_feature_volume), -1)
        # Computes the attention features by multiplying the anchor features with the attention weights per batch
        # This will give more context based on the probability of the current anchor to be a lane line compared to other frequently co-occurring anchor proposals
        # And adds them into a single tensor implicitly by using a matrix multiplication
        attention_features = torch.bmm(torch.transpose(batch_anchor_features, 1, 2),
                                       torch.transpose(attention_matrix, 1, 2)).transpose(1, 2)
        
        # Reshape the attention features batches to one batch size
        attention_features = attention_features.reshape(-1, self.__feature_volume_channels * self.__feature_volume_height)
        # Reshape the batch anchor features batches to one batch size
        batch_anchor_features = batch_anchor_features.reshape(-1, self.__feature_volume_channels * self.__feature_volume_height)

        # Concatenate the attention features with the anchor features
        batch_anchor_features = torch.cat((attention_features, batch_anchor_features), dim=1)

        # Predict the class of the anchor proposals
        cls_logits = self.__cls_layer(batch_anchor_features)
        # Predict the regression of the anchor proposals
        reg = self.__reg_layer(batch_anchor_features)

        # Undo joining the proposals from all images into proposals features batches
        cls_logits = cls_logits.reshape(x.shape[0], -1, cls_logits.shape[1])
        reg = reg.reshape(x.shape[0], -1, reg.shape[1])
        
        # Create the regression proposals
        reg_proposals = torch.zeros((*cls_logits.shape[:2], 5 + self.__anchor_y_steps), device=self.device)
        # Assign the anchor proposals to the regression proposals
        reg_proposals += self.__anchors_image
        # Assign the classification scores to the regression proposals
        reg_proposals[:, :, :2] = cls_logits
        # Adds the regression offsets to the anchor proposals in the regression proposals
        reg_proposals[:, :, 4:] += reg

    def __cut_anchor_features(self, feature_volumes):
        """
            Extracts anchor features from the feature volumes

            Args:
                feature_volumes (torch.Tensor): Feature volumes
        """

        # Gets the batch size
        batch_size = feature_volumes.shape[0]
        # Gets the number of anchor proposals
        anchor_proposals = len(self.__anchors_feature_volume)
        # Gets the number of channels in the feature volume
        feature_volume_channels = feature_volumes.shape[1]
        # Gets the height of the feature volume
        feature_volume_height = feature_volumes.shape[2]
        # Builds a tensor to store the anchor features for each batch
        batch_anchor_features = torch.zeros((batch_size, anchor_proposals, feature_volume_channels, feature_volume_height, 1), 
                                            device=self.device)
        
        # Iterates over each batch
        for batch_idx, feature_volume in enumerate(feature_volumes):
            # Extracts from each anchor proposal pixels in each feature map of the feature volume and transforms them into a tensor
            rois = feature_volume[self.__anchors_z_cut_indices, 
                                  self.__anchors_y_cut_indices, 
                                  self.__anchors_x_cut_indices].view(anchor_proposals, feature_volume_channels, feature_volume_height, 1)
            # Sets to zero the anchor proposals that are outside the feature map to avoid taking the edge values
            rois[self.__invalid_mask] = 0
            # Assigns the anchor features to the batch anchor features tensor
            batch_anchor_features[batch_idx] = rois

        return batch_anchor_features
    
    def train_model(self, resume=False):
        """
            Train the model
        """
        # Setup the logger
        logger = utils.setup_logging()
        logger.info('Starting training...')

        model = self.to(self.device)

        # Get the optimizer and the scheduler from the config file
        optimizer = getattr(torch.optim, self.__laneatt_config['optimizer']['name'])(model.parameters(), **self.__laneatt_config['optimizer']['parameters'])
        scheduler = getattr(torch.optim.lr_scheduler, self.__laneatt_config['lr_scheduler']['name'])(optimizer, **self.__laneatt_config['lr_scheduler']['parameters'])

        # State the starting epoch
        starting_epoch = 1
        # Load the last training state if the resume flag is set and modify the starting epoch and model
        if resume:
            last_epoch, model, optimizer, scheduler = utils.load_last_train_state(model, optimizer, scheduler)
            starting_epoch = last_epoch + 1
        
        epochs = self.__laneatt_config['epochs']
        train_loader = self.__get_dataloader('train')

    def __get_dataloader(self, split):
        # Create the dataset object based on TuSimple architecture
        train_dataset = utils.LaneDataset(self.__laneatt_config, split)
        # Create the dataloader object
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.__laneatt_config['batch_size'],
                                                   shuffle=True,
                                                   num_workers=20,
                                                   worker_init_fn=self.__worker_init_fn_)
        return train_loader

    @staticmethod
    def __worker_init_fn_(_):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)

if __name__ == '__main__':
    laneatt = LaneATT()
    laneatt(torch.randn(1, 3, 720, 1280))
    laneatt.train_model()