# coding: utf-8

from __future__ import division, print_function

import numpy as np
import tensorflow as tf
import cv2
import random

# 只针对一个样本的信息
def parse_line(line):
    '''
    Given a line from the training/test txt file, return parsed
    pic_path, boxes info, and label info.
    return:
        pic_path: string.
        boxes: shape [N, 4], N is the ground truth count, elements in the second
            dimension are [x_min, y_min, x_max, y_max]
    
    line = line.decode()
    s = line.strip().split(' ')
    pic_path = s[0]
    s = s[1:]
    box_cnt = len(s) // 5
    boxes = []
    labels = []
    for i in range(box_cnt):
        label, x_min, y_min, x_max, y_max = int(s[i*5]), float(s[i*5+1]), float(s[i*5+2]), float(s[i*5+3]), float(s[i*5+4])
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(label)
    boxes = np.asarray(boxes, np.float32)
    labels = np.asarray(labels, np.int64)
    return pic_path, boxes, labels
    '''
    if 'str' not in str(type(line)):
        line = line.decode()
    s = line.strip().split(' ')
    assert len(s) > 8, 'Annotation error! Please check your annotation file. Make sure there is at least one target object in each image.'
    line_idx = int(s[0])
    pic_path = s[1]
    img_width = int(s[2])
    img_height = int(s[3])
    s = s[4:]
    assert len(s) % 5 == 0, 'Annotation error! Please check your annotation file. Maybe partially missing some coordinates?'
    box_cnt = len(s) // 5
    boxes = []
    labels = []
    for i in range(box_cnt):

        label, x_min, y_min, x_max, y_max = int(s[i * 5]), float(s[i * 5 + 1]), float(s[i * 5 + 2]), float(
            s[i * 5 + 3]), float(s[i * 5 + 4])
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(label)
    boxes = np.asarray(boxes, np.float32)
    labels = np.asarray(labels, np.int64)
    return line_idx, pic_path, boxes, labels, img_width, img_height



def resize_image_and_correct_boxes(img, boxes, img_size):
    # convert gray scale image to 3-channel fake RGB image
    if len(img) == 2:
        img = np.expand_dims(img, -1)
    ori_height, ori_width = np.shape(img)[:2]
    new_width, new_height = img_size
    # shape to (new_height, new_width)
    img = cv2.resize(img, (new_width, new_height))

    # convert to float
    img = np.asarray(img, np.float32)
    
    # boxes
    # xmin, xmax
    boxes[:, 0] = boxes[:, 0] / ori_width * new_width
    boxes[:, 2] = boxes[:, 2] / ori_width * new_width
    # ymin, ymax
    boxes[:, 1] = boxes[:, 1] / ori_height * new_height
    boxes[:, 3] = boxes[:, 3] / ori_height * new_height

    return img, boxes


def data_augmentation(img, boxes, label):
    '''
    Do your own data augmentation here.
    note: if use the clip, the data_augmentation() must before the resize_image_and_correct_boxes()
    param:
        img: a [H, W, 3] shape RGB format image, float32 dtype
        boxes: [N, 4] shape boxes coordinate info, N is the ground truth box number,
            4 elements in the second dimension are [x_min, y_min, x_max, y_max], float32 dtype
        label: [N] shape labels, int64 dtype (you should not convert to int32)
    '''
    
    # randomly clip the image
    
    # img, boxes = clip_image(img, boxes)
    
    # randomly change the bright
    img = bright_image(img)
    
    # randomly change the saturation
    img = saturation_image(img)
    
    return img, boxes, label

# randomly clip the image
def clip_image(img, boxes):
    
    xmin = 10000
    ymin = 10000
    xmax = 0
    ymax = 0
    for i in range(len(boxes)): 
        if boxes[i][0] < xmin:
            xmin = boxes[i][0]
        
        if boxes[i][1] < ymin:
            ymin = boxes[i][1]
            
        if boxes[i][2] > xmax:
            xmax = boxes[i][2]
            
        if boxes[i][3] > ymax:
            ymax = boxes[i][3]
         
    max_h = int(np.shape(img)[0]/6)
    max_w = int(np.shape(img)[1]/6)
    rand_l = random.randint(0, max_w)
    rand_l = min(rand_l, xmin)
    rand_r = random.randint(0, max_w)
    rand_r = min(rand_r, np.shape(img)[1]-xmax)
    rand_u = random.randint(0, max_h)
    rand_u = min(rand_u, ymin)
    rand_d = random.randint(0, max_h)
    rand_d = min(rand_d, np.shape(img)[0]-ymax)
      
    img_clip = img[rand_u:np.shape(img)[0]-rand_d, rand_l:np.shape(img)[1]-rand_r]
    
    for i in range(len(boxes)):    
        boxes[i][0] = max(boxes[i][0]-rand_l, 0)
        boxes[i][1] = max(boxes[i][1]-rand_u, 0)
        boxes[i][2] = min(boxes[i][2]-rand_l, np.shape(img_clip)[1]-1)
        boxes[i][3] = min(boxes[i][3]-rand_u, np.shape(img_clip)[0]-1)

    return img_clip, boxes

# randomly change the bright
def bright_image(img, adjust=0.25):
    
    a = random.uniform(0, adjust)
    flag = random.randint(-1, 1)
           
    img_u = (255-img)*a
    img_d = img*a    
    
    delta = np.minimum(img_u, img_d)
    img_new = img + flag*delta
                        
    return img_new
  
    
# randomly change the saturation
def saturation_image(img, adjust=0.1):
           
    a_0 = random.uniform(1-adjust, 1+adjust)
    a_1 = random.uniform(1-adjust, 1+adjust)
    a_2 = random.uniform(1-adjust, 1+adjust)
    
    img_new = np.ones(img.shape) * 255
    img_new[:,:,0] = np.minimum(img[:,:,0] * a_0, img_new[:,:,0])
    img_new[:,:,1] = np.minimum(img[:,:,1] * a_1, img_new[:,:,1])
    img_new[:,:,2] = np.minimum(img[:,:,2] * a_2, img_new[:,:,2])
    
    return img_new
    

# 只针对一个样本的信息，成成 ground truth box 对应的网格信息，没有物体的网格值为 0
def process_box(boxes, labels, img_size, class_num, anchors):
    '''
    Generate the y_true label, i.e. the ground truth feature_maps in 3 different scales.
    '''
    anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]

    # convert boxes form:
    # shape: [N, 2]
    # (x_center, y_center)
    box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
    # (width, height)
    box_sizes = boxes[:, 2:4] - boxes[:, 0:2]

    # [13, 13, 3, 3+num_class]
    y_true_13 = np.zeros((img_size[1] // 32, img_size[0] // 32, 3, 5 + class_num), np.float32)
    y_true_26 = np.zeros((img_size[1] // 16, img_size[0] // 16, 3, 5 + class_num), np.float32)
    y_true_52 = np.zeros((img_size[1] // 8, img_size[0] // 8, 3, 5 + class_num), np.float32)

    y_true = [y_true_13, y_true_26, y_true_52]

    # [N, 1, 2]
    box_sizes = np.expand_dims(box_sizes, 1)
    # broadcast tricks
    # [N, 1, 2] & [9, 2] ==> [N, 9, 2]
    mins = np.maximum(- box_sizes / 2, - anchors / 2)
    maxs = np.minimum(box_sizes / 2, anchors / 2)
    # [N, 9, 2]
    whs = maxs - mins

    # [N, 9]  计算 ground truth box 与 9个 anchors 的 iou（因为每个像素都对应一个 anchor, 所以计算 iou 时不用考虑 x,y
    iou = (whs[:, :, 0] * whs[:, :, 1]) / (box_sizes[:, :, 0] * box_sizes[:, :, 1] + anchors[:, 0] * anchors[:, 1] - whs[:, :, 0] * whs[:, :, 1] + 1e-10)
    # [N] 找出 iou 最大的坐标
    best_match_idx = np.argmax(iou, axis=1)

    ratio_dict = {1.: 8., 2.: 16., 3.: 32.}
    for i, idx in enumerate(best_match_idx):
        # idx: 0,1,2 ==> 0; 3,4,5 ==> 1; 6,7,8 ==> 2  找出对应的尺寸
        feature_map_group = 2 - idx // 3
        # scale ratio: 0,1,2 ==> 8; 3,4,5 ==> 16; 6,7,8 ==> 32
        ratio = ratio_dict[np.ceil((idx + 1) / 3.)]
        x = int(np.floor(box_centers[i, 0] / ratio))    # 找出 ground truth box 对应的网格坐标
        y = int(np.floor(box_centers[i, 1] / ratio))
        k = anchors_mask[feature_map_group].index(idx)
        c = labels[i]
        # print feature_map_group, '|', y,x,k,c

        # 把信息写入对应的信息表中（成网格状），没有物体的网格信息为 0
        y_true[feature_map_group][y, x, k, :2] = box_centers[i]
        y_true[feature_map_group][y, x, k, 2:4] = box_sizes[i]
        y_true[feature_map_group][y, x, k, 4] = 1.
        y_true[feature_map_group][y, x, k, 5+c] = 1.

    return y_true_13, y_true_26, y_true_52

# 只针对一个样本的信息，成成 ground truth box 对应的网格信息，没有物体的网格值为 0
def process_box_tiny(boxes, labels, img_size, class_num, anchors):
    '''
    Generate the y_true label, i.e. the ground truth feature_maps in 3 different scales.
    '''
    anchors_mask = [[3,4,5], [0,1,2]]

    # convert boxes form:
    # shape: [N, 2]
    # (x_center, y_center)
    box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
    # (width, height)
    box_sizes = boxes[:, 2:4] - boxes[:, 0:2]

    # [13, 13, 3, 3+num_class]
    y_true_13 = np.zeros((img_size[1] // 32, img_size[0] // 32, 3, 5 + class_num), np.float32)
    y_true_26 = np.zeros((img_size[1] // 16, img_size[0] // 16, 3, 5 + class_num), np.float32)

    y_true = [y_true_13, y_true_26]

    # [N, 1, 2]
    box_sizes = np.expand_dims(box_sizes, 1)
    # broadcast tricks
    # [N, 1, 2] & [6, 2] ==> [N, 6, 2]
    mins = np.maximum(- box_sizes / 2, - anchors / 2)
    maxs = np.minimum(box_sizes / 2, anchors / 2)
    # [N, 6, 2]
    whs = maxs - mins

    # [N, 6]  计算 ground truth box 与 6个 anchors 的 iou（因为每个像素都对应一个 anchor, 所以计算 iou 时不用考虑 x,y
    iou = (whs[:, :, 0] * whs[:, :, 1]) / (box_sizes[:, :, 0] * box_sizes[:, :, 1] + anchors[:, 0] * anchors[:, 1] - whs[:, :, 0] * whs[:, :, 1] + 1e-10)
    # [N] 找出 iou 最大的坐标
    best_match_idx = np.argmax(iou, axis=1)

    ratio_dict = {1.: 16., 2.: 32.}
    for i, idx in enumerate(best_match_idx):
        # idx: 0,1,2 ==> 0; 3,4,5 ==> 1; 6,7,8 ==> 2  找出对应的尺寸
        feature_map_group = 1 - idx // 3
        # scale ratio: 0,1,2 ==> 8; 3,4,5 ==> 16; 6,7,8 ==> 32
        ratio = ratio_dict[np.ceil((idx + 1) / 3.)]
        x = int(np.floor(box_centers[i, 0] / ratio))    # 找出 ground truth box 对应的网格坐标
        y = int(np.floor(box_centers[i, 1] / ratio))
        k = anchors_mask[feature_map_group].index(idx)
        c = labels[i]
        # print feature_map_group, '|', y,x,k,c

        # 把信息写入对应的信息表中（成网格状），没有物体的网格信息为 0
        y_true[feature_map_group][y, x, k, :2] = box_centers[i]
        y_true[feature_map_group][y, x, k, 2:4] = box_sizes[i]
        y_true[feature_map_group][y, x, k, 4] = 1.
        y_true[feature_map_group][y, x, k, 5+c] = 1.

    return y_true_13, y_true_26


def parse_data(line, class_num, img_size, anchors, mode):
    '''
    param:
        line: a line from the training/test txt file
        args: args returned from the main program
        mode: 'train' or 'val'. When set to 'train', data_augmentation will be applied.
    '''
    _, pic_path, boxes, labels, _, _ = parse_line(line)

    img = cv2.imread(pic_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img, boxes = resize_image_and_correct_boxes(img, boxes, img_size)
    
    # do data augmentation here
    # note: if use the clip, the data_augmentation() must before the resize_image_and_correct_boxes()
    #if mode == 'train':
    #img, boxes, labels = data_augmentation(img, boxes, labels)

    # the input of yolo_v3 should be in range 0~1
    img = img / 255

    if (np.shape(anchors)[0] == 9):   
        y_true_13, y_true_26, y_true_52 = process_box(boxes, labels, img_size, class_num, anchors)   
        return img, y_true_13, y_true_26, y_true_52
    
    elif (np.shape(anchors)[0] == 6):   
        y_true_13, y_true_26 = process_box_tiny(boxes, labels, img_size, class_num, anchors)  
        y_true_xx = y_true_26
        return img, y_true_13, y_true_26, y_true_xx
