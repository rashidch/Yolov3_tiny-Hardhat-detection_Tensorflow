#  YOLOv3 and YOLOv3_tiny for TensorFlow

### 1. Introduction

Add YOLOv3_tiny and data augment(clip, brighten, change saturation)

### 2. Requirements

- tensorflow >= 1.8.0 (lower versions may work too)
- opencv-python


### 3. Running demos

(1) Single image test demo using ckpt file:

```shell
python test_single_image.py ./data/demo_data/car.jpg
```

(2) Single image test demo using pb file:

```shell
python test_single_image_pb.py ./data/demo_data/car.jpg
```

### 4. Training

#### 4.1 Data preparation 

(1) annotation file

Generate `train.txt/val.txt/test.txt` files under `./data/my_data/` directory. 
One line for one image, in the format like `image_absolute_path box_1 box_2 ... box_n`. 
Box_format: `label_index x_min y_min x_max y_max`.(The origin of coordinates is at the left top corner.)

For example:

```
xxx/xxx/1.jpg 0 453 369 473 391 1 588 245 608 268
xxx/xxx/2.jpg 1 466 403 485 422 2 793 300 809 320
...
```

**NOTE**: **You should leave a blank line at the end of each txt file.**

(2)  class_names file:

Generate the `data.names` file under `./data/my_data/` directory. Each line represents a class name.

For example:

```
bird
car
bike
...
```

The COCO dataset class names file is placed at `./data/coco.names`.

(3) prior anchor file:

Using the kmeans algorithm to get the prior anchors:

```
python get_kmeans.py
```

Then you will get 9 anchors and the average IOU. Save the anchors to a txt file.

The COCO dataset anchors offered by YOLO v3 author is placed at `./data/yolo_anchors.txt`, you can use that one too.

**NOTE: The yolo anchors should be scaled to the rescaled new image size. 
Suppose your image size is [W, H], and the image will be rescale to 416*416 as input, for each generated anchor [anchor_w, anchor_h], 
you should apply the transformation anchor_w = anchor_w / W * 416, anchor_h = anchor_g / H * 416.**

#### 4.2 Training

Using `train.py`. The parameters are as following:

```shell
$ python train.py -h
usage: train.py 

        net_name = 'the yolo model'
        anchors_name = 'the anchors name'
        body_name = 'the yolo body net'
        data_name = 'the training data name'


```

Check the `train.py` for more details. You should set the parameters yourself. 

Some training tricks in my experiment:

the yolov3 using  `darknet53`, the yolov3_tiny using `darknet19`



### Credits:

I refer to many fantastic repos during the implementation:

https://github.com/wizyoung/YOLOv3_TensorFlow





 