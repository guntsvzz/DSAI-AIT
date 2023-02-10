import torch
import numpy as np
import torch.nn as nn
from util import *
import math

def run_training(model, optimizer, dataloader, device, img_size, n_epoch, every_n_batch, every_n_epoch, ckpt_dir):
    losses = None
    for epoch_i in range(n_epoch):
        running_loss = 0.0
        for inputs, labels, bboxes in dataloader:
            inputs = torch.from_numpy(np.array(inputs)).squeeze(1).permute(0,3,1,2).float()
            inputs = inputs.to(device)
            labels = torch.stack(labels).to(device)
            #print(inputs.shape)

            running_corrects = 0

            # zero the parameter gradients
            # it uses for update training weights
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs, True)

                pred_xywh = outputs[..., 0:4] / img_size
                pred_conf = outputs[..., 4:5]
                pred_cls = outputs[..., 5:]

                label_xywh = labels[..., :4] / img_size

                label_obj_mask = labels[..., 4:5]
                label_noobj_mask = (1.0 - label_obj_mask)
                lambda_coord = 0.001
                lambda_noobj = 0.05
                label_cls = labels[..., 5:]
                loss = nn.MSELoss()
                loss_bce = nn.BCELoss()

                loss_coord = lambda_coord * label_obj_mask * loss(input=pred_xywh, target=label_xywh)
                loss_conf = (label_obj_mask * loss_bce(input=pred_conf, target=label_obj_mask)) + \
                            (lambda_noobj * label_noobj_mask * loss_bce(input=pred_conf, target=label_obj_mask))
                loss_cls = label_obj_mask * loss_bce(input=pred_cls, target=label_cls)

                loss_coord = torch.sum(loss_coord)
                loss_conf = torch.sum(loss_conf)
                loss_cls = torch.sum(loss_cls)

                # print(pred_xywh.shape, label_xywh.shape)

                ciou = CIOU_xywh_torch(pred_xywh, label_xywh)
                # print(ciou.shape)
                ciou = ciou.unsqueeze(-1)
                # print(ciou.shape)
                # print(label_obj_mask.shape)
                loss_ciou = torch.sum(label_obj_mask * (1.0 - ciou))
                # print(loss_coord)
                loss =  loss_ciou +  loss_conf + loss_cls
                loss.backward()
                optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                # print('Running loss')
                # print(loss_coord, loss_conf, loss_cls)
        epoch_loss = running_loss / 750
        print(epoch_loss)
        print('End Epoch')

def iou_xywh_numpy(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    # print(boxes1, boxes2)

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU


def CIOU_xywh_torch(boxes1,boxes2):
    '''
    cal CIOU of two boxes or batch boxes
    :param boxes1:[xmin,ymin,xmax,ymax] or
                [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
    :param boxes2:[xmin,ymin,xmax,ymax]
    :return:
    '''
    # cx cy w h->xyxy
    boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
    boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)

    boxes1 = torch.cat([torch.min(boxes1[..., :2], boxes1[..., 2:]),
                        torch.max(boxes1[..., :2], boxes1[..., 2:])], dim=-1)
    boxes2 = torch.cat([torch.min(boxes2[..., :2], boxes2[..., 2:]),
                        torch.max(boxes2[..., :2], boxes2[..., 2:])], dim=-1)

    # (x2 minus x1 = width)  * (y2 - y1 = height)
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # upper left of the intersection region (x,y)
    inter_left_up = torch.max(boxes1[..., :2], boxes2[..., :2])

    # bottom right of the intersection region (x,y)
    inter_right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    # if there is overlapping we will get (w,h) else set to (0,0) because it could be negative if no overlapping
    inter_section = torch.max(inter_right_down - inter_left_up, torch.zeros_like(inter_right_down))
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = 1.0 * inter_area / union_area

    # cal outer boxes
    outer_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
    outer_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    outer = torch.max(outer_right_down - outer_left_up, torch.zeros_like(inter_right_down))
    outer_diagonal_line = torch.pow(outer[..., 0], 2) + torch.pow(outer[..., 1], 2)

    # cal center distance
    # center x center y
    boxes1_center = (boxes1[..., :2] +  boxes1[...,2:]) * 0.5
    boxes2_center = (boxes2[..., :2] +  boxes2[...,2:]) * 0.5

    # euclidean distance
    # x1-x2 square 
    center_dis = torch.pow(boxes1_center[...,0]-boxes2_center[...,0], 2) +\
                 torch.pow(boxes1_center[...,1]-boxes2_center[...,1], 2)

    # cal penalty term
    # cal width,height
    boxes1_size = torch.max(boxes1[..., 2:] - boxes1[..., :2], torch.zeros_like(inter_right_down))
    boxes2_size = torch.max(boxes2[..., 2:] - boxes2[..., :2], torch.zeros_like(inter_right_down))
    v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan((boxes1_size[...,0]/torch.clamp(boxes1_size[...,1],min = 1e-6))) -
            torch.atan((boxes2_size[..., 0] / torch.clamp(boxes2_size[..., 1],min = 1e-6))), 2)

    alpha = v / (1-ious+v)

    #cal ciou
    cious = ious - (center_dis / outer_diagonal_line + alpha*v)

    return cious