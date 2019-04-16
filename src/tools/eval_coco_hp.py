from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import sys
import cv2
import numpy as np
import pickle
import os

this_dir = os.path.dirname(__file__)
ANN_PATH = this_dir + '../../data/coco/annotations/person_keypoints_val2017.json'
print(ANN_PATH)
if __name__ == '__main__':
  pred_path = sys.argv[1]
  coco = coco.COCO(ANN_PATH)
  dets = coco.loadRes(pred_path)
  img_ids = coco.getImgIds()
  num_images = len(img_ids)
  coco_eval = COCOeval(coco, dets, "keypoints")
  coco_eval.evaluate()
  coco_eval.accumulate()
  coco_eval.summarize()
  coco_eval = COCOeval(coco, dets, "bbox")
  coco_eval.evaluate()
  coco_eval.accumulate()
  coco_eval.summarize()
  
