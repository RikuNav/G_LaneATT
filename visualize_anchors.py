import torch
import matplotlib.pyplot as plt
import numpy as np

from model.laneatt import LaneATT
from PIL import Image

laneatt = LaneATT()

anchors = laneatt.generate_anchors(lateral_n=72, bottom_n=128)[0]

img = Image.open('model/inv2.jpg')
img_width, img_height = img.size

plt.imshow(img)
plt.axis('off')

for i, proposal in enumerate(anchors):
    if i % 21 != 0:
        continue
    height_steps = len(proposal)
    proposal = torch.clamp(proposal, 0, img_width)
    y_points = np.linspace(img_height, 0, height_steps)
    plt.plot(proposal, y_points, 'r')

plt.show()