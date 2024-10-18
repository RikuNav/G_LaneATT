import torch
import numpy as np

def generate_anchors(lateral_n, bottom_n, left_angles, right_angles, bottom_angles, y_steps, feature_volume_height):
    """
        Generates anchors for the model

        Args:
            lateral_n (int): Number of lateral anchors
            bottom_n (int): Number of bottom anchors
            left_angles (list): List of left angles
            right_angles (list): List of right angles
            bottom_angles (list): List of bottom angles
            y_steps (int): Number of steps in y direction
            feature_volume_height (int): Height of the feature map
            img_w (int): Image width

        Returns:
            torch.Tensor: Anchors for all sides
            torch.Tensor: Cut anchors for all sides
    """

    # Generate left anchors
    left_anchors, left_cut = generate_side_anchors(left_angles, lateral_n, y_steps, feature_volume_height, x=0.)
    # Generate right anchors
    right_anchors, right_cut = generate_side_anchors(right_angles, lateral_n, y_steps, feature_volume_height, x=1.)
    # Generate bottom anchors
    bottom_anchors, bottom_cut = generate_side_anchors(bottom_angles, bottom_n, y_steps, feature_volume_height, y=1.)

    # Concatenate anchors and cut anchors
    return torch.cat([left_anchors, bottom_anchors, right_anchors]), torch.cat([left_cut, bottom_cut, right_cut]) 

def generate_side_anchors(angles, nb_origins, y_steps, feature_volume_height, x=None, y=None,):
    """
        Generates side anchors based on predefined angles, number of origins and coordinate

        Args:
            angles (list): List of angles
            nb_origins (int): Number of origins
            y_steps (int): Number of steps in y direction
            feature_volume_height (int): Height of the feature map
            img_w (int): Image width
            x (float): X coordinate
            y (float): Y coordinate

        Returns:
            torch.Tensor: Anchors for the side
            torch.Tensor: Cut anchors for the side
    """

    # Check if x or y is None
    if x is None and y is not None:
        # Generate starts based on a fixed y
        starts = [(x, y) for x in np.linspace(1., 0., num=nb_origins)]
    elif x is not None and y is None:
        # Generate starts based on a fixed x
        starts = [(x, y) for y in np.linspace(1., 0., num=nb_origins)]
    else:
        # Raises an error if no side is defined
        raise Exception('Please define exactly one of `x` or `y` (not neither nor both)')

    # Calculate number of anchors dince one anchor is generated for each angle and origin
    n_anchors = nb_origins * len(angles)

    # Initialize anchors and cut anchors as a tensor of n_anchors as rows and (offsets or feature map height + 5) as columns.
    # This represents each anchor list will have 2 scores, 1 start y, 1 start x, 1 length and n_offsets or feature map height
    anchors = torch.zeros((n_anchors, 2 + 2 + 1 + y_steps))
    anchors_cut = torch.zeros((n_anchors, 2 + 2 + 1 + feature_volume_height))

    # Iterates over each start point
    for i, start in enumerate(starts):
        # Iterates over each angle for each start point
        for j, angle in enumerate(angles):
            # Calculates the index of the anchor
            k = i * len(angles) + j
            # Generates the anchor and cut anchor
            anchors[k] = generate_anchor(start, angle, y_steps, feature_volume_height,)
            anchors_cut[k] = generate_anchor(start, angle, y_steps, feature_volume_height, cut=True)

    return anchors, anchors_cut

def generate_anchor(start, angle, y_steps, feature_volume_height, cut=False):
    """
        Generates anchor based on start point and angle

        Args:
            start (tuple): Start point
            angle (float): Angle
            y_steps (int): Number of steps in y direction
            feature_volume_height (int): Height of the feature map
            img_w (int): Image width
            cut (bool): If cut

        Returns:
            torch.Tensor: Anchor
    """

    # Check if cut is True
    if cut:
        # Set anchor y coordinates from 1 to 0 with feature map height steps
        anchor_ys = torch.linspace(1, 0, steps=feature_volume_height, dtype=torch.float32)
        # Initialize anchor tensor with 2 scores, 1 start y, 1 start x, 1 length and feature map height
        anchor = torch.zeros(2 + 2 + 1 + feature_volume_height)
    else:
        # Set anchor y coordinates from 1 to 0 with n_offsets steps
        anchor_ys = torch.linspace(1, 0, steps=y_steps, dtype=torch.float32)
        # Initialize anchor tensor with 2 scores, 1 start y, 1 start x, 1 length and n_offsets
        anchor = torch.zeros(2 + 2 + 1 + y_steps)
    # Convert angle to radians
    angle = angle * np.pi / 180.
    # Extract start x and y from start point
    start_x, start_y = start
    # Assigns to third element of anchor tensor the start y taking the bottom as 0
    anchor[2] = 1 - start_y
    # Assigns to fourth element of anchor tensor the start x
    anchor[3] = start_x
    # Gets a relative delta y based on the start coordinate for each n_offsets points
    delta_y = (anchor_ys - start_y)
    # Gets a relative delta x from the origin point for each anchor point based on the angle and delta y since -> 1/tan(angle) = delta x / delta y
    delta_x = delta_y / np.tan(angle)
    # Adds the delta x of each anchor point to the start x to get the x coordinate of each anchor point
    anchor[5:] = start_x + delta_x

    return anchor

def compute_anchor_cut_indices(anchors_feature_volume, feature_map_channels, feature_volume_height, feature_volume_width):
        """
            Computes anchor cut indices

            Args:
                anchors_cut (torch.Tensor): Cut anchors
                feature_map_channels (int): Number of feature map channels
                feature_volume_height (int): Height of the feature map
                feature_volume_width (int): Width of the feature map
            
            Returns:
                torch.Tensor: Z coordinates
                torch.Tensor: Y coordinates
                torch.Tensor: X coordinates
        """
        # Get the number of anchors proposals
        n_proposals = len(anchors_feature_volume)

        # Extract only anchor points from anchors_feature_volume tensor
        anchors_x_points = anchors_feature_volume[:, 5:]
        # Remap points from 1-0 range to feature map pixels
        anchors_x_pixels = (anchors_x_points * feature_volume_width).round().long()
        # Flip the x coordinates to get from feature_volume_width->0 to 0->feature_volume_width and adds a third dimension
        anchors_x_pixels = torch.flip(anchors_x_pixels, dims=(1,)).unsqueeze(2)
        # Repeat the anchors proposals for each feature map and puts them in a single dimension
        unclamped_anchors_x_cut_indices = torch.repeat_interleave(anchors_x_pixels, feature_map_channels, dim=0).reshape(-1, 1)
        # Clamp the anchors coordinates to the feature map width
        anchors_x_cut_indices = torch.clamp(unclamped_anchors_x_cut_indices, 0, feature_volume_width - 1)
        # Reshape the anchors to the original shape
        unclamped_anchors_x_cut_indices = unclamped_anchors_x_cut_indices.reshape(n_proposals, feature_map_channels, feature_volume_height, 1)
        
        # Generate a binary mask to filter out the invalid anchor proposals
        invalid_mask = (unclamped_anchors_x_cut_indices < 0) | (unclamped_anchors_x_cut_indices > feature_volume_width - 1)

        # Generate y coordinates for each anchor point
        anchors_y_cut_indices = torch.arange(0, feature_volume_height)
        # Repeat the y coordinates for each feature map and each proposal and puts them in a single dimension
        anchors_y_cut_indices = anchors_y_cut_indices.repeat(feature_map_channels).repeat(n_proposals).reshape(-1, 1)

        # Generate z coordinates for each anchor proposal and puts them in a single dimension
        anchors_z_cut_indices = torch.arange(feature_map_channels).repeat_interleave(feature_volume_height).repeat(n_proposals).reshape(-1, 1)

        return anchors_z_cut_indices, anchors_y_cut_indices, anchors_x_cut_indices, invalid_mask