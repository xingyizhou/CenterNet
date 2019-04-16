import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import sys
import cv2
import numpy as np
import pickle
IMG_PATH = '../../data/coco/val2017/'
ANN_PATH = '../../data/coco/annotations/instances_val2017.json'
DEBUG = True

def _coco_box_to_bbox(box):
  bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                  dtype=np.int32)
  return bbox

_cat_ids = [
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
  14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
  24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
  37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
  48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
  58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
  72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
  82, 84, 85, 86, 87, 88, 89, 90
]
num_classes = 80
_classes = {
  ind + 1: cat_id for ind, cat_id in enumerate(_cat_ids)
}
_to_order = {cat_id: ind for ind, cat_id in enumerate(_cat_ids)}
coco = coco.COCO(ANN_PATH)
CAT_NAMES = [coco.loadCats([_classes[i + 1]])[0]['name'] \
              for i in range(num_classes)]
COLORS = [((np.random.random((3, )) * 0.6 + 0.4)*255).astype(np.uint8) \
              for _ in range(num_classes)]


def add_box(image, bbox, sc, cat_id):
  cat_id = _to_order[cat_id]
  cat_name = CAT_NAMES[cat_id]
  cat_size  = cv2.getTextSize(cat_name + '0', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
  color = np.array(COLORS[cat_id]).astype(np.int32).tolist()
  txt = '{}{:.0f}'.format(cat_name, sc * 10)
  if bbox[1] - cat_size[1] - 2 < 0:
    cv2.rectangle(image,
                  (bbox[0], bbox[1] + 2),
                  (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 2),
                  color, -1)
    cv2.putText(image, txt, 
                (bbox[0], bbox[1] + cat_size[1] + 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)
  else:
    cv2.rectangle(image,
                  (bbox[0], bbox[1] - cat_size[1] - 2),
                  (bbox[0] + cat_size[0], bbox[1] - 2),
                  color, -1)
    cv2.putText(image, txt, 
                (bbox[0], bbox[1] - 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)
  cv2.rectangle(image,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color, 2)
  return image

if __name__ == '__main__':
  dets = []
  img_ids = coco.getImgIds()
  num_images = len(img_ids)
  for k in range(1, len(sys.argv)):
    pred_path = sys.argv[k]
    dets.append(coco.loadRes(pred_path))
  # import pdb; pdb.set_trace()
  for i, img_id in enumerate(img_ids):
    img_info = coco.loadImgs(ids=[img_id])[0]
    img_path = IMG_PATH + img_info['file_name']
    img = cv2.imread(img_path)
    gt_ids = coco.getAnnIds(imgIds=[img_id])
    gts = coco.loadAnns(gt_ids)
    gt_img = img.copy()
    for j, pred in enumerate(gts):
      bbox = _coco_box_to_bbox(pred['bbox'])
      cat_id = pred['category_id']
      gt_img = add_box(gt_img, bbox, 0, cat_id)
    for k in range(len(dets)):
      pred_ids = dets[k].getAnnIds(imgIds=[img_id])
      preds = dets[k].loadAnns(pred_ids)
      pred_img = img.copy()
      for j, pred in enumerate(preds):
        bbox = _coco_box_to_bbox(pred['bbox'])
        sc = pred['score']
        cat_id = pred['category_id']
        if sc > 0.2:
          pred_img = add_box(pred_img, bbox, sc, cat_id)
      cv2.imshow('pred{}'.format(k), pred_img)
      # cv2.imwrite('vis/{}_pred{}.png'.format(i, k), pred_img)
    cv2.imshow('gt', gt_img)
    # cv2.imwrite('vis/{}_gt.png'.format(i), gt_img)
    cv2.waitKey()
  # coco_eval.evaluate()
  # coco_eval.accumulate()
  # coco_eval.summarize()

  
