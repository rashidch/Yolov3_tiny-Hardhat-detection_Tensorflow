# coding: utf-8

from __future__ import division, print_function

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

def conv2d(inputs, filters, kernel_size, strides=1):
    def _fixed_padding(inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
        return padded_inputs
    if strides > 1: 
        inputs = _fixed_padding(inputs, kernel_size)
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides,
                         padding=('SAME' if strides == 1 else 'VALID'))
    return inputs

def darknet53_body(inputs):
    
    def res_block(inputs, filters):
        shortcut = inputs
        net = conv2d(inputs, filters * 1, 1)
        net = conv2d(net, filters * 2, 3)

        net = net + shortcut

        return net
    
    # first two conv2d layers
    net = conv2d(inputs, 32,  3, strides=1)
    net = conv2d(net, 64,  3, strides=2)

    # res_block * 1
    net = res_block(net, 32)

    net = conv2d(net, 128, 3, strides=2)

    # res_block * 2
    for i in range(2):
        net = res_block(net, 64)

    net = conv2d(net, 256, 3, strides=2)

    # res_block * 8
    for i in range(8):
        net = res_block(net, 128)

    route_1 = net
    net = conv2d(net, 512, 3, strides=2)

    # res_block * 8
    for i in range(8):
        net = res_block(net, 256)

    route_2 = net
    net = conv2d(net, 1024, 3, strides=2)

    # res_block * 4
    for i in range(4):
        net = res_block(net, 512)
    route_3 = net

    return route_1, route_2, route_3


def darknet19_body(inputs):
    net = conv2d(inputs, 16,  3, strides=1)   
    net = slim.max_pool2d(net, [2,2], stride=2, padding='SAME')
    # 208 x 208
    net = conv2d(net, 32,  3, strides=1)
    net = slim.max_pool2d(net, [2,2], stride=2, padding='SAME')
    # 104 x 104
    net = conv2d(net, 64,  3, strides=1)
    net = slim.max_pool2d(net, [2,2], stride=2, padding='SAME')
    route_1f = net
    # 52 x 52
    net = conv2d(net, 128,  3, strides=1)
    net = slim.max_pool2d(net, [2,2], stride=2, padding='SAME')   
    route_1 = net
    # 26 x 26
    net = conv2d(net, 256,  3, strides=1)
    route_2f = net
    net = slim.max_pool2d(net, [2,2], stride=2, padding='SAME')
    # 13 x 13
    net = conv2d(net, 512,  3, strides=1)
    net = slim.max_pool2d(net, [2,2], stride=1, padding='SAME')
    # 13 x 13
    net = conv2d(net, 1024,  3, strides=1)
    route_2 = net
    # 13 x 13
    return route_1, route_1f, route_2, route_2f


def yolo_block(inputs, filters):
    net = conv2d(inputs, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = conv2d(net, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = conv2d(net, filters * 1, 1)
    route = net
    net = conv2d(net, filters * 2, 3)
    return route, net

def yolo_tiny_block(inputs, filters):
    net = conv2d(inputs, filters * 1, 1)
    route = net
    net = conv2d(net, filters * 2, 3)
    return route, net


def upsample_layer(inputs, out_shape):
    new_height, new_width = out_shape[1], out_shape[2]
    # NOTE: here height is the first
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), align_corners=True, name='upsampled')
    return inputs

def PAM_layer(net):
    
    N, H, W, C = np.shape(net)
    
    net_b = slim.conv2d(net, C, 1, stride=1)
    net_c = slim.conv2d(net, C, 1, stride=1)
    net_d = slim.conv2d(net, C, 1, stride=1)
    
    net_b = tf.reshape(net_b, [-1, H*W, C])
    net_c = tf.reshape(net_c, [-1, H*W, C])
    net_c = tf.transpose(net_c, [0,2,1])
    net_d = tf.reshape(net_d, [-1, H*W, C])
    
    net_bc = tf.matmul(net_b,net_c)
    net_bcd = tf.matmul(net_bc,net_d)
    
    net_bcd = tf.reshape(net_bcd, [-1, H, W, C])
    
    net = net_bcd * 0.2 + net
    
    return net


