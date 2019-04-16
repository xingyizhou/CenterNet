from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import pycocotools.coco as coco
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
import pycocotools.coco as coco
import math

class EXDetDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def __getitem__(self, index):
    img_id = self.images[index]
    img_info = self.coco.loadImgs(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    img = cv2.imread(img_path)

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.])
    s = max(img.shape[0], img.shape[1]) * 1.0
    
    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:
        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
      else:
        sf = self.opt.scale
        cf = self.opt.shift
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
        c[0] += img.shape[1] * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += img.shape[0] * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]

    trans_input = get_affine_transform(
      c, s, 0, [self.opt.input_res, self.opt.input_res])
    inp = cv2.warpAffine(img, trans_input, 
                         (self.opt.input_res, self.opt.input_res),
                         flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.)
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    output_res = self.opt.output_res
    num_classes = self.opt.num_classes
    trans_output = get_affine_transform(c, s, 0, [output_res, output_res])
    num_hm = 1 if self.opt.agnostic_ex else num_classes

    hm_t = np.zeros((num_hm, output_res, output_res), dtype=np.float32)
    hm_l = np.zeros((num_hm, output_res, output_res), dtype=np.float32)
    hm_b = np.zeros((num_hm, output_res, output_res), dtype=np.float32)
    hm_r = np.zeros((num_hm, output_res, output_res), dtype=np.float32)
    hm_c = np.zeros((num_classes, output_res, output_res), dtype=np.float32)
    reg_t = np.zeros((self.max_objs, 2), dtype=np.float32)
    reg_l = np.zeros((self.max_objs, 2), dtype=np.float32)
    reg_b = np.zeros((self.max_objs, 2), dtype=np.float32)
    reg_r = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind_t = np.zeros((self.max_objs), dtype=np.int64)
    ind_l = np.zeros((self.max_objs), dtype=np.int64)
    ind_b = np.zeros((self.max_objs), dtype=np.int64)
    ind_r = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs)
    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian

    for k in range(num_objs):
      ann = anns[k]
      # bbox = self._coco_box_to_bbox(ann['bbox'])
      # tlbr
      pts = np.array(ann['extreme_points'], dtype=np.float32).reshape(4, 2)
      # cls_id = int(self.cat_ids[ann['category_id']] - 1) # bug
      cls_id = int(self.cat_ids[ann['category_id']])
      hm_id = 0 if self.opt.agnostic_ex else cls_id
      if flipped:
        pts[:, 0] = width - pts[:, 0] - 1
        pts[1], pts[3] = pts[3].copy(), pts[1].copy()
      for j in range(4):
        pts[j] = affine_transform(pts[j], trans_output)
      pts = np.clip(pts, 0, self.opt.output_res - 1)
      h, w = pts[2, 1] - pts[0, 1], pts[3, 0] - pts[1, 0]
      if h > 0 and w > 0:
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        pt_int = pts.astype(np.int32)
        draw_gaussian(hm_t[hm_id], pt_int[0], radius)
        draw_gaussian(hm_l[hm_id], pt_int[1], radius)
        draw_gaussian(hm_b[hm_id], pt_int[2], radius)
        draw_gaussian(hm_r[hm_id], pt_int[3], radius)
        reg_t[k] = pts[0] - pt_int[0]
        reg_l[k] = pts[1] - pt_int[1]
        reg_b[k] = pts[2] - pt_int[2]
        reg_r[k] = pts[3] - pt_int[3]
        ind_t[k] = pt_int[0, 1] * output_res + pt_int[0, 0]
        ind_l[k] = pt_int[1, 1] * output_res + pt_int[1, 0]
        ind_b[k] = pt_int[2, 1] * output_res + pt_int[2, 0]
        ind_r[k] = pt_int[3, 1] * output_res + pt_int[3, 0]

        ct = [int((pts[3, 0] + pts[1, 0]) / 2), int((pts[0, 1] + pts[2, 1]) / 2)]
        draw_gaussian(hm_c[cls_id], ct, radius)
        reg_mask[k] = 1
    ret = {'input': inp, 'hm_t': hm_t, 'hm_l': hm_l, 'hm_b': hm_b, 
            'hm_r': hm_r, 'hm_c': hm_c}
    if self.opt.reg_offset:
      ret.update({'reg_mask': reg_mask,
        'reg_t': reg_t, 'reg_l': reg_l, 'reg_b': reg_b, 'reg_r': reg_r,
        'ind_t': ind_t, 'ind_l': ind_l, 'ind_b': ind_b, 'ind_r': ind_r})
    
    return ret