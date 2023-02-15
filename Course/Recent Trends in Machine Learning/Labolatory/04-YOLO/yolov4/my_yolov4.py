# %%
from darknet import *
import torch
import cv2

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

model = Darknet("cfg/yolov4.cfg")
model.module_list[114].conv_114 = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
# model.load_weights("csdarknet53-omega_final.weights",backbone = True)
input = get_test_input()
print(input.shape)
prediction = model(input, False)
# print(prediction)
print(prediction.shape)
# %%
# model.load_weights("yolov4.weights")
input = get_test_input()
prediction = model(input, False)
write_results(prediction.detach(), 0.5, 80, nms_conf = 0.4)
# print(prediction)
print(prediction.shape)
num_classes = 80
classes = load_classes("../data/coco.names")

print(classes)
# %%
