#%%
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from torchvision.models import resnet50
import torch
import cv2
import numpy as np

from resnet_model import ResNet18
import torch
# Set device to GPU or CPU

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

resnet = ResNet18().to(device)
resnet.load_state_dict(torch.load('resnet18SGD_WD_bestsofar.pth'))

# Set device to GPU or CPU

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

model = resnet
print(model)

#%%
target_layers = [model.layer4[-1]]

# rgb_img = cv2.imread("example01.jpeg", 1)[:, :, ::-1]
rgb_img = cv2.imread('./PASCALVOC2012/2008_000012.jpeg', 1)[:, :, ::-1]
rgb_img = np.float32(rgb_img) / 255
print(rgb_img.shape)
input_tensor = preprocess_image(rgb_img,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

print(input_tensor.shape)
# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

# You can also use it within a with statement, to make sure it is freed,
# In case you need to re-create it inside an outer loop:
# with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
#   ...

# We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.

# 281: tabby, tabby cat
# 229: Old English sheepdog, bobtail
targets = [ClassifierOutputTarget(0), ClassifierOutputTarget(1), ClassifierOutputTarget(2), ClassifierOutputTarget(3), ClassifierOutputTarget(4),
        ClassifierOutputTarget(5), ClassifierOutputTarget(6), ClassifierOutputTarget(7), ClassifierOutputTarget(8)]

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
print(grayscale_cam.shape)
print(rgb_img.shape)
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# %%
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import deprocess_image

cam_image = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
gb = gb_model(input_tensor, target_category=None)

print(gb.shape)

cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
cam_gb = deprocess_image(cam_mask * gb)
gb = deprocess_image(gb)

cv2.imwrite(f'check_ex_cam.jpg', cam_image)
# cv2.imwrite(f'ex_gb.jpg', gb)
# cv2.imwrite(f'ex_cam_gb.jpg', cam_gb)