#  Hardhat (Helmet) detection from construction site using YOLOv3_tiny with TensorFlow

### 1. Introduction

Hardhat detection using Yolov3_tiny

### 2. Requirements

- tensorflow >= 1.8.0 (lower versions may work too)
- opencv-python


### 3. Running demos

#### (1) Single image test demo using ckpt file:

```shell
python test_single_image.py ./data/demo_data/google_image.jpg
```

#### test image from google
![detection result](./detection%20results/test_result_google_image.jpg)

#### test image from dataset
![detection result](./detection%20results/16.jpg)





#### (2) Video demo: https://drive.google.com/drive/folders/1YirwBUWwvecjgk-MwDXk1hZaJJGJSEJ_?usp=sharing

### 4. Training

Hardhat dataset: pascal voc format: https://drive.google.com/drive/folders/12WtXQyM-7jWvWPtCXZlnycsIK72ClHgu?usp=sharing

Dataset credits: https://github.com/wujixiu/helmet-detection

#### 4.1 Data preparation 

(1) annotation file

Generate `train.txt/val.txt/test.txt` files under `./data/my_data/` directory. 
One line for one image, in the format like `image_absolute_path image size box_1 box_2 ... box_n`. 
Box_format: `label_index x_min y_min x_max y_max`.(The origin of coordinates is at the left top corner.)

For example:

```
577 /home/rashid/YOLOv3_TensorFlow-master/data/my_data/GDUT-HWD/JPEGImages/01457.jpg 440 293 1 235 84 258 110 1 291 93 307 115 1 320 96 335 114
743 /home/rashid/YOLOv3_TensorFlow-master/data/my_data/GDUT-HWD/JPEGImages/00179.jpg 1300 956 1 503 80 674 313 1 258 1 423 222
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

The code is inspired from following repos :

https://github.com/wizyoung/YOLOv3_TensorFlow

https://github.com/Huangdebo/YOLOv3_tiny_TensorFlow

Dataset credits: https://github.com/wujixiu/helmet-detection





 
