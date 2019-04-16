from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import cv2
import sys
import time
from utils.debugger import Debugger
from models.data_parallel import DataParallel
from models.losses import FocalLoss, RegL1Loss
from models.decode import agnex_ct_decode, exct_decode
from models.utils import _sigmoid
from .base_trainer import BaseTrainer

class ExdetLoss(torch.nn.Module):
  def __init__(self, opt):
    super(ExdetLoss, self).__init__()
    self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_reg = RegL1Loss()
    self.opt = opt
    self.parts = ['t', 'l', 'b', 'r', 'c']

  def forward(self, outputs, batch):
    opt = self.opt
    hm_loss, reg_loss = 0, 0
    for s in range(opt.num_stacks):
      output = outputs[s]
      for p in self.parts:
        tag = 'hm_{}'.format(p)
        output[tag] = _sigmoid(output[tag])
        hm_loss += self.crit(output[tag], batch[tag]) / opt.num_stacks
        if p != 'c' and opt.reg_offset and opt.off_weight > 0:
          reg_loss += self.crit_reg(output['reg_{}'.format(p)], 
                                    batch['reg_mask'],
                                    batch['ind_{}'.format(p)],
                                    batch['reg_{}'.format(p)]) / opt.num_stacks
    loss = opt.hm_weight * hm_loss + opt.off_weight * reg_loss
    loss_stats = {'loss': loss, 'off_loss': reg_loss, 'hm_loss': hm_loss}
    return loss, loss_stats

class ExdetTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(ExdetTrainer, self).__init__(opt, model, optimizer=optimizer)
    self.decode = agnex_ct_decode if opt.agnostic_ex else exct_decode

  def _get_losses(self, opt):
    loss_states = ['loss', 'hm_loss', 'off_loss']
    loss = ExdetLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt
    detections = self.decode(output['hm_t'], output['hm_l'], 
                             output['hm_b'], output['hm_r'], 
                             output['hm_c']).detach().cpu().numpy()
    detections[:, :, :4] *= opt.input_res / opt.output_res
    for i in range(1):
      debugger = Debugger(
        dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
      pred_hm = np.zeros((opt.input_res, opt.input_res, 3), dtype=np.uint8)
      gt_hm = np.zeros((opt.input_res, opt.input_res, 3), dtype=np.uint8)
      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.opt.std + self.opt.mean) * 255.).astype(np.uint8)
      for p in self.parts:
        tag = 'hm_{}'.format(p)
        pred = debugger.gen_colormap(output[tag][i].detach().cpu().numpy())
        gt = debugger.gen_colormap(batch[tag][i].detach().cpu().numpy())
        if p != 'c':
          pred_hm = np.maximum(pred_hm, pred)
          gt_hm = np.maximum(gt_hm, gt)
        if p == 'c' or opt.debug > 2:
          debugger.add_blend_img(img, pred, 'pred_{}'.format(p))
          debugger.add_blend_img(img, gt, 'gt_{}'.format(p))
      debugger.add_blend_img(img, pred_hm, 'pred')
      debugger.add_blend_img(img, gt_hm, 'gt')
      debugger.add_img(img, img_id='out')
      for k in range(len(detections[i])):
        if detections[i, k, 4] > 0.1:
          debugger.add_coco_bbox(detections[i, k, :4], detections[i, k, -1],
                                 detections[i, k, 4], img_id='out')
      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)