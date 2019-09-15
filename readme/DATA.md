# Dataset preparation

If you want to reproduce the results in the paper for benchmark evaluation and training, you will need to setup dataset.


### COCO
- Download the images (2017 Train, 2017 Val, 2017 Test) from [coco website](http://cocodataset.org/#download).
- Download annotation files (2017 train/val and test image info) from [coco website](http://cocodataset.org/#download). 
- Place the data (or create symlinks) to make the data folder like:

  ~~~
  ${CenterNet_ROOT}
  |-- data
  `-- |-- coco
      `-- |-- annotations
          |   |-- instances_train2017.json
          |   |-- instances_val2017.json
          |   |-- person_keypoints_train2017.json
          |   |-- person_keypoints_val2017.json
          |   |-- image_info_test-dev2017.json
          |---|-- train2017
          |---|-- val2017
          `---|-- test2017
  ~~~

- [Optional] If you want to train ExtremeNet, generate extreme point annotation from segmentation:
    
    ~~~
    cd $CenterNet_ROOT/tools/
    python gen_coco_extreme_points.py
    ~~~
  It generates `instances_extreme_train2017.json` and `instances_extreme_val2017.json` in `data/coco/annotations/`. 

### Pascal VOC

- Run

    ~~~
    cd $CenterNet_ROOT/tools/
    bash get_pascal_voc.sh
    ~~~
- The above script includes:
    - Download, unzip, and move Pascal VOC images from the [VOC website](http://host.robots.ox.ac.uk/pascal/VOC/). 
    - [Download](https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip) Pascal VOC annotation in COCO format (from [Detectron](https://github.com/facebookresearch/Detectron/tree/master/detectron/datasets/data)). 
    - Combine train/val 2007/2012 annotation files into a single json. 


- Move the created `voc` folder to `data` (or create symlinks) to make the data folder like:

  ~~~
  ${CenterNet_ROOT}
  |-- data
  `-- |-- voc
      `-- |-- annotations
          |   |-- pascal_trainval0712.json
          |   |-- pascal_test2017.json
          |-- images
          |   |-- 000001.jpg
          |   ......
          `-- VOCdevkit
  
  ~~~
  The `VOCdevkit` folder is needed to run the evaluation script from [faster rcnn](https://github.com/rbgirshick/py-faster-rcnn/blob/master/tools/reval.py).

### KITTI

- Download [images](http://www.cvlibs.net/download.php?file=data_object_image_2.zip), [annotations](http://www.cvlibs.net/download.php?file=data_object_label_2.zip), and [calibrations](http://www.cvlibs.net/download.php?file=data_object_calib.zip) from [KITTI website](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and unzip.

- Download the train-val split of [3DOP](https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz) and [SubCNN](https://github.com/tanshen/SubCNN/tree/master/fast-rcnn/data/KITTI) and place the data as below

  ~~~
  ${CenterNet_ROOT}
  |-- data
  `-- |-- kitti
      `-- |-- training
          |   |-- image_2
          |   |-- label_2
          |   |-- calib
          |-- ImageSets_3dop
          |   |-- test.txt
          |   |-- train.txt
          |   |-- val.txt
          |   |-- trainval.txt
          `-- ImageSets_subcnn
              |-- test.txt
              |-- train.txt
              |-- val.txt
              |-- trainval.txt
  ~~~

- Run `python convert_kitti_to_coco.py` in `tools` to convert the annotation into COCO format. You can set `DEBUG=True` in `line 5` to visualize the annotation.

- Link image folder

  ~~~
  cd ${CenterNet_ROOT}/data/kitti/
  mkdir images
  ln -s training/image_2 images/trainval
  ~~~

- The data structure should look like:

  ~~~
  ${CenterNet_ROOT}
  |-- data
  `-- |-- kitti
      `-- |-- annotations
          |   |-- kitti_3dop_train.json
          |   |-- kitti_3dop_val.json
          |   |-- kitti_subcnn_train.json
          |   |-- kitti_subcnn_val.json
          `-- images
              |-- trainval
              |-- test
  ~~~
