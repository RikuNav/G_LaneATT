import os
import json
import random
import logging
import numpy as np
import cv2

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from scipy.interpolate import InterpolatedUnivariateSpline

SPLIT_FILES = {
    'train+val': ['label_data_0313.json', 'label_data_0601.json', 'label_data_0531.json'],
    'train': ['label_data_0313.json', 'label_data_0601.json'],
    'val': ['label_data_0531.json'],
    'test': ['test_label.json'],
}

class LaneDataset(Dataset):
    def __init__(self, config, split):
        self.__general_config = config
        self.__dataset_config = config['dataset'][split]
        # Split means the dataset split to load, e.g. train, val, test
        self.__split = split
        # The root directory of the dataset
        self.__root = os.path.join(os.path.dirname(os.path.dirname(__file__)), self.__dataset_config['root'])
        # A logger object
        self.__logger = logging.getLogger(__name__)

        self.img_w, self.img_h = self.__general_config['image_size']['width'], self.__general_config['image_size']['height']

        # Verify that the split exists
        if split not in SPLIT_FILES.keys():
            raise Exception('Split `{}` does not exist.'.format(split))

        # Load the annotation files
        self.__annotation_files = [os.path.join(self.__root, path) for path in SPLIT_FILES[split]]

        # Verify that the root directory is specified
        if self.__root is None:
            raise Exception('Please specify the root directory')

        self.__annotations = []
        self.__load_annotations()

        self.__logger.info("Transforming annotations to the model's target format...")
        self.__y_steps = self.__general_config['anchor_steps']['y']
        self.__offsets_ys = np.arange(self.__img_h, -1, -self.img_h / (self.__y_steps-1))
        self.annotations = np.array(list(map(self.__transform_annotation, self.__annotations)))
        self.__logger.info("Annotations transformed.")

    @property
    def annotations(self):
        return self.__annotations
    
    @annotations.setter
    def annotations(self, annotations):
        self.__annotations = annotations
    
    @property
    def img_w(self):
        return self.__img_w
    
    @img_w.setter
    def img_w(self, img_w):
        self.__img_w = img_w
    
    @property
    def img_h(self):
        return self.__img_h
    
    @img_h.setter
    def img_h(self, img_h):
        self.__img_h = img_h

    @property
    def max_lanes(self):
        return self.__max_lanes
    
    @max_lanes.setter
    def max_lanes(self, max_lanes):
        self.__max_lanes = max_lanes

    def __load_annotations(self):
        self.__logger.info('Loading TuSimple annotations...')
        max_lanes = 0
        # Iterate over the annotation files
        for annotation_file in self.__annotation_files:
            # Opens and reads the annotation file
            with open(annotation_file, 'r') as annotation_obj:
                lines = annotation_obj.readlines()
            # Iterate over the lines in the annotation file
            for line in lines:
                # Load the JSON line data
                data = json.loads(line)
                # Get the lanes y coordinates
                y_samples = data['h_samples']
                # Get the lanes x coordinates
                og_lanes = data['lanes']
                # Create the lanes as a tuple of x and y coordinates
                lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in og_lanes]
                lanes = [lane for lane in lanes if len(lane) > 0]
                max_lanes = max(max_lanes, len(lanes))

                # Append the annotation to the list of annotations
                self.__annotations.append({
                    'path': os.path.join(self.__root, data['raw_file']),
                    'org_path': data['raw_file'],
                    'org_lanes': og_lanes,
                    'lanes': lanes,
                    'aug': False,
                    'y_samples': y_samples
                })

        # Shuffle the annotations if the split is train
        if self.__split == 'train':
            random.shuffle(self.__annotations)
        self.max_lanes = max_lanes
        self.__logger.info('%d annotations loaded, with a maximum of %d lanes in an image.', len(self.__annotations),
                         self.__max_lanes)

    def __transform_annotation(self, annotation):
        old_lanes = annotation['lanes']

        # Remove lanes with less than 2 points
        old_lanes = filter(lambda x: len(x) > 1, old_lanes)
        # Sort lane points by Y (bottom to top of the image)
        old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]
        # Remove points with same Y (keep first occurrence)
        old_lanes = [self.__filter_lane(lane) for lane in old_lanes]
        # Create tranformed annotations
        lanes = np.ones((self.max_lanes, 2 + 1 + 1 + 1 + self.__y_steps),
                        dtype=np.float32) * -1e5
        # Lanes are invalid by default
        lanes[:, 0] = 1
        lanes[:, 1] = 0

        # Iterates over the lanes
        for lane_idx, lane in enumerate(old_lanes):
            try:
                xs_outside_image, xs_inside_image = self.__sample_lane(lane, self.__offsets_ys)
            except AssertionError:
                continue
            if len(xs_inside_image) == 0:
                continue
            # Joins the points outside and inside the image again
            all_xs = np.hstack((xs_outside_image, xs_inside_image))
            # Updates the lanes with the new points
            # Sets the scores to 0 for no lane and 1 for lane
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            # Set y starting point in 1-0 range
            lanes[lane_idx, 2] = len(xs_outside_image) / (self.__y_steps-1)
            # Set x starting point
            lanes[lane_idx, 3] = xs_inside_image[0]
            # Set the number of points in the lane
            lanes[lane_idx, 4] = len(xs_inside_image)
            # Set the x coordinates of the lane
            lanes[lane_idx, 5:5 + len(all_xs)] = all_xs

        new_annotation = {'path': annotation['path'], 'label': lanes, 'old_annotation': annotation}
        return new_annotation
    
    def __filter_lane(self, lane):
        """
            Remove points with the same Y coordinate, keeping only the first occurrence.

            Args:
                lane: A list of points representing a lane.

            Returns:
                A list of points representing a lane with the same Y coordinate removed.
        """
        assert lane[-1][1] <= lane[0][1] # Invalid lane
        filtered_lane = []
        used = set()
        for p in lane:
            if p[1] not in used:
                filtered_lane.append(p)
                used.add(p[1])

        return filtered_lane
    
    def __sample_lane(self, points, sample_ys):
        """
            Sample the lane points at the anchor points.

            Args:
                points: A list of points representing a lane.
                sample_ys: The anchor points to sample the lane points.
            
            Returns:    
                A tuple containing the points outside the image and the points inside the image.
        """
        # Verify that the points are sorted
        points = np.array(points)
        if not np.all(points[1:, 1] < points[:-1, 1]):
            raise Exception('Annotaion points have to be sorted')
        x, y = points[:, 0], points[:, 1]

        # Interpolate points inside domain
        assert len(points) > 1
        interp = InterpolatedUnivariateSpline(y[::-1], x[::-1], k=min(3, len(points) - 1))
        domain_min_y = y.min()
        domain_max_y = y.max()
        # Sample points from the anchor y steps inside the domain of the dataset
        sample_ys_inside_domain = sample_ys[(sample_ys >= domain_min_y) & (sample_ys <= domain_max_y)]
        assert len(sample_ys_inside_domain) > 0
        # Evaluates the y samples in the interpolation function to get the x samples
        interp_xs = interp(sample_ys_inside_domain)

        # Extrapolate lane with a straight line using the 2 points closest to the bottom
        two_closest_points = points[:2]
        extrap = np.polyfit(two_closest_points[:, 1], two_closest_points[:, 0], deg=1)
        extrap_ys = sample_ys[sample_ys > domain_max_y]
        extrap_xs = np.polyval(extrap, extrap_ys)
        all_xs = np.hstack((extrap_xs, interp_xs))

        # Separate between inside and outside points
        inside_mask = (all_xs >= 0) & (all_xs < self.img_w)
        xs_inside_image = all_xs[inside_mask]
        xs_outside_image = all_xs[~inside_mask]

        return xs_outside_image, xs_inside_image

    def __getitem__(self, idx):
        item = self.annotations[idx]
        img_org = cv2.imread(item['path'])
        img = ToTensor()((img_org.copy()/255.0).astype(np.float32))
        label = item['label']
        return (img, label)
    
    def __len__(self):
        return len(self.__annotations)
    
