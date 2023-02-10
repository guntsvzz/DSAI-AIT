from __future__ import division
import time
import os
import os.path as osp
import numpy as np
import cv2
import pickle as pkl
import pandas as pd
import random
from copy import copy, deepcopy
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Subset,DataLoader
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision import datasets, models, transforms

from util_default import *

import albumentations as A

# Set device to GPU or CPU
gpu = "2"
device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")


path2data_train="/path/to/train/images/"
path2json_train="/path/to/train/coco_annotation.json"

path2data_val="/path/to/validate/images/"
path2json_val="/path/to/validate/coco_annotation.json"

img_size = 416

train_transform = A.Compose([
    #A.SmallestMaxSize(256),
    A.Resize(img_size, img_size),
    # A.RandomCrop(width=224, height=224),
    # A.HorizontalFlip(p=0.5),
    # A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
)

eval_transform = A.Compose([
    A.Resize(img_size, img_size),
    #A.SmallestMaxSize(256),
    #A.CenterCrop(width=224, height=224),
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
)

BATCH_SIZE = 10
def collate_fn(batch):
    return tuple(zip(*batch))

from cococustom import CustomCoco

train_dataset = Subset(CustomCoco(root = path2data_train,
                                annFile = path2json_train, transform=train_transform), list(range(0,20)))
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=0, collate_fn=collate_fn)

from darknet_default import Darknet

print("Loading network.....")
model = Darknet("cfg/yolov3.cfg")
# load pretrained
# model.load_weights("yolov3.weights")
print("Network successfully loaded")