from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss, RegL1Loss, RegLoss, RegWeightedL1Loss
from models.decode import multi_pose_decode
from models.utils import _sigmoid, flip_tensor, flip_lr_off, flip_lr
from utils.debugger import Debugger
from utils.post_process import multi_pose_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer

class MultiPoseLoss(torch.nn.Module):
  def __init__(self, opt):
    super(MultiPoseLoss, self).__init__()
    self.crit = FocalLoss()
    self.crit_hm_hp = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_kp = RegWeightedL1Loss() if not opt.dense_hp else \
                   torch.nn.L1Loss(reduction='sum')
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
                    RegLoss() if opt.reg_loss == 'sl1' else None
    self.opt = opt

  def forward(self, outputs, batch):
    opt = self.opt
    hm_loss, wh_loss, off_loss = 0, 0, 0
    hp_loss, off_loss, hm_hp_loss, hp_offset_loss = 0, 0, 0, 0
    for s in range(opt.num_stacks):
      output = outputs[s]
      output['hm'] = _sigmoid(output['hm'])
      if opt.hm_hp and not opt.mse_loss:
        output['hm_hp'] = _sigmoid(output['hm_hp'])
      
      if opt.eval_oracle_hmhp:
        output['hm_hp'] = batch['hm_hp']
      if opt.eval_oracle_hm:
        output['hm'] = batch['hm']
      if opt.eval_oracle_kps:
        if opt.dense_hp:
          output['hps'] = batch['dense_hps']
        else:
          output['hps'] = torch.from_numpy(gen_oracle_map(
            batch['hps'].detach().cpu().numpy(), 
            batch['ind'].detach().cpu().numpy(), 
            opt.output_res, opt.output_res)).to(opt.device)
      if opt.eval_oracle_hp_offset:
        output['hp_offset'] = torch.from_numpy(gen_oracle_map(
          batch['hp_offset'].detach().cpu().numpy(), 
          batch['hp_ind'].detach().cpu().numpy(), 
          opt.output_res, opt.output_res)).to(opt.device)


      hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
      if opt.dense_hp:
        mask_weight = batch['dense_hps_mask'].sum() + 1e-4
        hp_loss += (self.crit_kp(output['hps'] * batch['dense_hps_mask'], 
                                 batch['dense_hps'] * batch['dense_hps_mask']) / 
                                 mask_weight) / opt.num_stacks
      else:
        hp_loss += self.crit_kp(output['hps'], batch['hps_mask'], 
                                batch['ind'], batch['hps']) / opt.num_stacks
      if opt.wh_weight > 0:
        wh_loss += self.crit_reg(output['wh'], batch['reg_mask'],
                                 batch['ind'], batch['wh']) / opt.num_stacks
      if opt.reg_offset and opt.off_weight > 0:
        off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                  batch['ind'], batch['reg']) / opt.num_stacks
      if opt.reg_hp_offset and opt.off_weight > 0:
        hp_offset_loss += self.crit_reg(
          output['hp_offset'], batch['hp_mask'],
          batch['hp_ind'], batch['hp_offset']) / opt.num_stacks
      if opt.hm_hp and opt.hm_hp_weight > 0:
        hm_hp_loss += self.crit_hm_hp(
          output['hm_hp'], batch['hm_hp']) / opt.num_stacks
    loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
           opt.off_weight * off_loss + opt.hp_weight * hp_loss + \
           opt.hm_hp_weight * hm_hp_loss + opt.off_weight * hp_offset_loss
    
    loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'hp_loss': hp_loss, 
                  'hm_hp_loss': hm_hp_loss, 'hp_offset_loss': hp_offset_loss,
                  'wh_loss': wh_loss, 'off_loss': off_loss}
    return loss, loss_stats

class MultiPoseTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(MultiPoseTrainer, self).__init__(opt, model, optimizer=optimizer)
  
  def _get_losses(self, opt):
    loss_states = ['loss', 'hm_loss', 'hp_loss', 'hm_hp_loss', 
                   'hp_offset_loss', 'wh_loss', 'off_loss']
    loss = MultiPoseLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt
    reg = output['reg'] if opt.reg_offset else None
    hm_hp = output['hm_hp'] if opt.hm_hp else None
    hp_offset = output['hp_offset'] if opt.reg_hp_offset else None
    dets = multi_pose_decode(
      output['hm'], output['wh'], output['hps'], 
      reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])

    dets[:, :, :4] *= opt.input_res / opt.output_res
    dets[:, :, 5:39] *= opt.input_res / opt.output_res
    dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
    dets_gt[:, :, :4] *= opt.input_res / opt.output_res
    dets_gt[:, :, 5:39] *= opt.input_res / opt.output_res
    for i in range(1):
      debugger = Debugger(
        dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')

      debugger.add_img(img, img_id='out_pred')
      for k in range(len(dets[i])):
        if dets[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                 dets[i, k, 4], img_id='out_pred')
          debugger.add_coco_hp(dets[i, k, 5:39], img_id='out_pred')

      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt[i])):
        if dets_gt[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                 dets_gt[i, k, 4], img_id='out_gt')
          debugger.add_coco_hp(dets_gt[i, k, 5:39], img_id='out_gt')

      if opt.hm_hp:
        pred = debugger.gen_colormap_hp(output['hm_hp'][i].detach().cpu().numpy())
        gt = debugger.gen_colormap_hp(batch['hm_hp'][i].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hmhp')
        debugger.add_blend_img(img, gt, 'gt_hmhp')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)

  def save_result(self, output, batch, results):
    reg = output['reg'] if self.opt.reg_offset else None
    hm_hp = output['hm_hp'] if self.opt.hm_hp else None
    hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
    dets = multi_pose_decode(
      output['hm'], output['wh'], output['hps'], 
      reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    
    dets_out = multi_pose_post_process(
      dets.copy(), batch['meta']['c'].cpu().numpy(),
      batch['meta']['s'].cpu().numpy(),
      output['hm'].shape[2], output['hm'].shape[3])
    results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]