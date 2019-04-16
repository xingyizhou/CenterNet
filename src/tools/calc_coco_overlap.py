from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as COCO
import cv2
import numpy as np
from pycocotools import mask as maskUtils
ANN_PATH = '../../data/coco/annotations/'
IMG_PATH = '../../data/coco/'
ANN_FILES = {'train': 'instances_train2017.json',
             'val': 'instances_val2017.json'}
DEBUG = False
RESIZE = True

class_name = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def iou(box1, box2):
  area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
  area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
  inter = max(min(box1[2], box2[2]) - max(box1[0], box2[0]) + 1, 0) * \
          max(min(box1[3], box2[3]) - max(box1[1], box2[1]) + 1, 0)
  iou = 1.0 * inter / (area1 + area2 - inter)
  return iou

def generate_anchors(
    stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)
):
    """Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    are centered on stride / 2, have (approximate) sqrt areas of the specified
    sizes, and aspect ratios as given.
    """
    return _generate_anchors(
        stride,
        np.array(sizes, dtype=np.float) / stride,
        np.array(aspect_ratios, dtype=np.float)
    )


def _generate_anchors(base_size, scales, aspect_ratios):
    """Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    """
    anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
    anchors = _ratio_enum(anchor, aspect_ratios)
    anchors = np.vstack(
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
    )
    return anchors


def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1)
        )
    )
    return anchors


def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _coco_box_to_bbox(box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

def count_agnostic(split):
  coco = COCO.COCO(ANN_PATH + ANN_FILES[split])
  images = coco.getImgIds()
  cnt = 0
  for img_id in images:
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ids=ann_ids)
    centers = []
    for ann in anns:
      bbox = ann['bbox']
      center = ((bbox[0] + bbox[2] / 2) // 4, (bbox[1] + bbox[3] / 2) // 4)
      for c in centers:
        if center[0] == c[0] and center[1] == c[1]:
          cnt += 1
      centers.append(center)
  print('find {} collisions!'.format(cnt))


def count(split):
  coco = COCO.COCO(ANN_PATH + ANN_FILES[split])
  images = coco.getImgIds()
  cnt = 0
  obj = 0
  for img_id in images:
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ids=ann_ids)
    centers = []
    obj += len(anns)
    for ann in anns:
      if ann['iscrowd'] > 0:
        continue
      bbox = ann['bbox']
      center = ((bbox[0] + bbox[2] / 2) // 4, (bbox[1] + bbox[3] / 2) // 4, ann['category_id'], bbox)
      for c in centers:
        if center[0] == c[0] and center[1] == c[1] and center[2] == c[2] and \
           iou(_coco_box_to_bbox(bbox), _coco_box_to_bbox(c[3])) < 2:# 0.5:
          cnt += 1
          if DEBUG:
            file_name = coco.loadImgs(ids=[img_id])[0]['file_name']
            img = cv2.imread('{}/{}2017/{}'.format(IMG_PATH, split, file_name))
            x1, y1 = int(c[3][0]), int(c[3][1]), 
            x2, y2 = int(c[3][0] + c[3][2]), int(c[3][1] + c[3][3]) 
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA)
            x1, y1 = int(center[3][0]), int(center[3][1]), 
            x2, y2 = int(center[3][0] + center[3][2]), int(center[3][1] + center[3][3]) 
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('img', img)
            cv2.waitKey()
      centers.append(center)
  print('find {} collisions of {} objects!'.format(cnt, obj))

def count_iou(split):
  coco = COCO.COCO(ANN_PATH + ANN_FILES[split])
  images = coco.getImgIds()
  cnt = 0
  obj = 0
  for img_id in images:
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ids=ann_ids)
    bboxes = []
    obj += len(anns)
    for ann in anns:
      if ann['iscrowd'] > 0:
        continue
      bbox = _coco_box_to_bbox(ann['bbox']).tolist() + [ann['category_id']]
      for b in bboxes:
        if iou(b, bbox) > 0.5 and b[4] == bbox[4]:
          cnt += 1
          if DEBUG:
            file_name = coco.loadImgs(ids=[img_id])[0]['file_name']
            img = cv2.imread('{}/{}2017/{}'.format(IMG_PATH, split, file_name))
            x1, y1 = int(b[0]), int(b[1]), 
            x2, y2 = int(b[2]), int(b[3]) 
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA)
            x1, y1 = int(bbox[0]), int(bbox[1]), 
            x2, y2 = int(bbox[2]), int(bbox[3]) 
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('img', img)
            print('cats', class_name[b[4]], class_name[bbox[4]])
            cv2.waitKey()
      bboxes.append(bbox)
  print('find {} collisions of {} objects!'.format(cnt, obj))


def count_anchor(split):
  coco = COCO.COCO(ANN_PATH + ANN_FILES[split])
  images = coco.getImgIds()
  cnt = 0
  obj = 0
  stride = 16
  anchor = generate_anchors().reshape(15, 2, 2)
  miss_s, miss_m, miss_l = 0, 0, 0
  N = len(images)
  print(N, 'images')
  for ind, img_id in enumerate(images):
    if ind % 1000 == 0:
      print(ind, N)
    anchors = []
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ids=ann_ids)
    obj += len(anns)
    img_info = coco.loadImgs(ids=[img_id])[0]
    h, w = img_info['height'], img_info['width']
    if RESIZE:
      if h > w:
        for i in range(len(anns)):
          anns[i]['bbox'][0] *= 800 / w
          anns[i]['bbox'][1] *= 800 / w
          anns[i]['bbox'][2] *= 800 / w
          anns[i]['bbox'][3] *= 800 / w
        h = h * 800 // w
        w = 800 
      else:
        for i in range(len(anns)):
          anns[i]['bbox'][0] *= 800 / h
          anns[i]['bbox'][1] *= 800 / h
          anns[i]['bbox'][2] *= 800 / h
          anns[i]['bbox'][3] *= 800 / h
        w = w * 800 // h
        h = 800 
    for i in range(w // stride):
      for j in range(h // stride):
        ct = np.array([i * stride, j * stride], dtype=np.float32).reshape(1, 1, 2)
        anchors.append(anchor + ct)
    anchors = np.concatenate(anchors, axis=0).reshape(-1, 4)
    anchors[:, 2:4] = anchors[:, 2:4] - anchors[:, 0:2]
    anchors = anchors.tolist()
    # import pdb; pdb.set_trace()
    g = [g['bbox'] for g in anns]
    iscrowd = [int(o['iscrowd']) for o in anns]
    ious = maskUtils.iou(anchors,g,iscrowd)
    for t in range(len(g)):
      if ious[:, t].max() < 0.5:
        s = anns[t]['area']
        if s < 32 ** 2:
          miss_s += 1
        elif s < 96 ** 2:
          miss_m += 1
        else:
          miss_l += 1
    if DEBUG:
      file_name = coco.loadImgs(ids=[img_id])[0]['file_name']
      img = cv2.imread('{}/{}2017/{}'.format(IMG_PATH, split, file_name))
      if RESIZE:
        img = cv2.resize(img, (w, h))
      for t, gt in enumerate(g):
        if anns[t]['iscrowd'] > 0:
          continue
        x1, y1, x2, y2 = _coco_box_to_bbox(gt)
        cl = (0, 0, 255) if ious[:, t].max() < 0.5 else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), cl, 2, cv2.LINE_AA)
        for k in range(len(anchors)):
          if ious[k, t] > 0.5:
            x1, y1, x2, y2 = _coco_box_to_bbox(anchors[k])
            cl = (np.array([255, 0, 0]) * ious[k, t]).astype(np.int32).tolist()
            cv2.rectangle(img, (x1, y1), (x2, y2), cl, 1, cv2.LINE_AA)
      cv2.imshow('img', img)
      cv2.waitKey()
    miss = 0
    if len(ious) > 0:
      miss = (ious.max(axis=0) < 0.5).sum()
    cnt += miss
  print('cnt, obj, ratio ', cnt, obj, cnt / obj)
  print('s, m, l ', miss_s, miss_m, miss_l)
    # import pdb; pdb.set_trace()


def count_size(split):
  coco = COCO.COCO(ANN_PATH + ANN_FILES[split])
  images = coco.getImgIds()
  cnt = 0
  obj = 0
  stride = 16
  anchor = generate_anchors().reshape(15, 2, 2)
  cnt_s, cnt_m, cnt_l = 0, 0, 0
  N = len(images)
  print(N, 'images')
  for ind, img_id in enumerate(images):
    anchors = []
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ids=ann_ids)
    obj += len(anns)
    img_info = coco.loadImgs(ids=[img_id])[0]
    for t in range(len(anns)):
      if 1:
        s = anns[t]['area']
        if s < 32 ** 2:
          cnt_s += 1
        elif s < 96 ** 2:
          cnt_m += 1
        else:
          cnt_l += 1
      cnt += 1
  print('cnt', cnt)
  print('s, m, l ', cnt_s, cnt_m, cnt_l)
 

# count_iou('train')
# count_anchor('train')
# count('train')
count_size('train')





