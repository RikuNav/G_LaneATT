import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model.laneatt import LaneATT
from PIL import Image

laneatt = LaneATT('resnet50', anchor_feat_channels=3)

transform = transforms.Compose([transforms.Resize((640, 320)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])

img = Image.open('model/1.jpg')
img_tensor = transform(img).unsqueeze(0)

feature_volume = laneatt.forward(img_tensor).squeeze().permute(1, 2, 0).detach().numpy()

plt.imshow(feature_volume)
plt.axis('off')
plt.show()