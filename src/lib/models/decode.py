from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _gather_feat, _tranpose_and_gather_feat

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _left_aggregate(heat):
    '''
        heat: batchsize x channels x h x w
    '''
    shape = heat.shape 
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(1, heat.shape[0]):
        inds = (heat[i] >= heat[i - 1])
        ret[i] += ret[i - 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape) 

def _right_aggregate(heat):
    '''
        heat: batchsize x channels x h x w
    '''
    shape = heat.shape 
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(heat.shape[0] - 2, -1, -1):
        inds = (heat[i] >= heat[i +1])
        ret[i] += ret[i + 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape) 

def _top_aggregate(heat):
    '''
        heat: batchsize x channels x h x w
    '''
    heat = heat.transpose(3, 2) 
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(1, heat.shape[0]):
        inds = (heat[i] >= heat[i - 1])
        ret[i] += ret[i - 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape).transpose(3, 2)

def _bottom_aggregate(heat):
    '''
        heat: batchsize x channels x h x w
    '''
    heat = heat.transpose(3, 2) 
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(heat.shape[0] - 2, -1, -1):
        inds = (heat[i] >= heat[i + 1])
        ret[i] += ret[i + 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape).transpose(3, 2)

def _h_aggregate(heat, aggr_weight=0.1):
    return aggr_weight * _left_aggregate(heat) + \
           aggr_weight * _right_aggregate(heat) + heat

def _v_aggregate(heat, aggr_weight=0.1):
    return aggr_weight * _top_aggregate(heat) + \
           aggr_weight * _bottom_aggregate(heat) + heat

'''
# Slow for large number of categories
def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs
'''
def _topk_channel(scores, K=40):
      batch, cat, height, width = scores.size()
      
      topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

      topk_inds = topk_inds % (height * width)
      topk_ys   = (topk_inds / width).int().float()
      topk_xs   = (topk_inds % width).int().float()

      return topk_scores, topk_inds, topk_ys, topk_xs

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def agnex_ct_decode(
    t_heat, l_heat, b_heat, r_heat, ct_heat, 
    t_regr=None, l_regr=None, b_regr=None, r_regr=None, 
    K=40, scores_thresh=0.1, center_thresh=0.1, aggr_weight=0.0, num_dets=1000
):
    batch, cat, height, width = t_heat.size()

    '''
    t_heat  = torch.sigmoid(t_heat)
    l_heat  = torch.sigmoid(l_heat)
    b_heat  = torch.sigmoid(b_heat)
    r_heat  = torch.sigmoid(r_heat)
    ct_heat = torch.sigmoid(ct_heat)
    '''
    if aggr_weight > 0: 
      t_heat = _h_aggregate(t_heat, aggr_weight=aggr_weight)
      l_heat = _v_aggregate(l_heat, aggr_weight=aggr_weight)
      b_heat = _h_aggregate(b_heat, aggr_weight=aggr_weight)
      r_heat = _v_aggregate(r_heat, aggr_weight=aggr_weight)
      
    # perform nms on heatmaps
    t_heat = _nms(t_heat)
    l_heat = _nms(l_heat)
    b_heat = _nms(b_heat)
    r_heat = _nms(r_heat)
      
      
    t_heat[t_heat > 1] = 1
    l_heat[l_heat > 1] = 1
    b_heat[b_heat > 1] = 1
    r_heat[r_heat > 1] = 1

    t_scores, t_inds, _, t_ys, t_xs = _topk(t_heat, K=K)
    l_scores, l_inds, _, l_ys, l_xs = _topk(l_heat, K=K)
    b_scores, b_inds, _, b_ys, b_xs = _topk(b_heat, K=K)
    r_scores, r_inds, _, r_ys, r_xs = _topk(r_heat, K=K)
      
    ct_heat_agn, ct_clses = torch.max(ct_heat, dim=1, keepdim=True)
      
    # import pdb; pdb.set_trace()

    t_ys = t_ys.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    t_xs = t_xs.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    l_ys = l_ys.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    l_xs = l_xs.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    b_ys = b_ys.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    b_xs = b_xs.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    r_ys = r_ys.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
    r_xs = r_xs.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)

    box_ct_xs = ((l_xs + r_xs + 0.5) / 2).long()
    box_ct_ys = ((t_ys + b_ys + 0.5) / 2).long()

    ct_inds     = box_ct_ys * width + box_ct_xs
    ct_inds     = ct_inds.view(batch, -1)
    ct_heat_agn = ct_heat_agn.view(batch, -1, 1)
    ct_clses    = ct_clses.view(batch, -1, 1)
    ct_scores   = _gather_feat(ct_heat_agn, ct_inds)
    clses       = _gather_feat(ct_clses, ct_inds)

    t_scores = t_scores.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    l_scores = l_scores.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    b_scores = b_scores.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    r_scores = r_scores.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
    ct_scores = ct_scores.view(batch, K, K, K, K)
    scores    = (t_scores + l_scores + b_scores + r_scores + 2 * ct_scores) / 6

    # reject boxes based on classes
    top_inds  = (t_ys > l_ys) + (t_ys > b_ys) + (t_ys > r_ys)
    top_inds = (top_inds > 0)
    left_inds  = (l_xs > t_xs) + (l_xs > b_xs) + (l_xs > r_xs)
    left_inds = (left_inds > 0)
    bottom_inds  = (b_ys < t_ys) + (b_ys < l_ys) + (b_ys < r_ys)
    bottom_inds = (bottom_inds > 0)
    right_inds  = (r_xs < t_xs) + (r_xs < l_xs) + (r_xs < b_xs)
    right_inds = (right_inds > 0)

    sc_inds = (t_scores < scores_thresh) + (l_scores < scores_thresh) + \
              (b_scores < scores_thresh) + (r_scores < scores_thresh) + \
              (ct_scores < center_thresh)
    sc_inds = (sc_inds > 0)

    scores = scores - sc_inds.float()
    scores = scores - top_inds.float()
    scores = scores - left_inds.float()
    scores = scores - bottom_inds.float()
    scores = scores - right_inds.float()

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)

    if t_regr is not None and l_regr is not None \
      and b_regr is not None and r_regr is not None:
        t_regr = _tranpose_and_gather_feat(t_regr, t_inds)
        t_regr = t_regr.view(batch, K, 1, 1, 1, 2)
        l_regr = _tranpose_and_gather_feat(l_regr, l_inds)
        l_regr = l_regr.view(batch, 1, K, 1, 1, 2)
        b_regr = _tranpose_and_gather_feat(b_regr, b_inds)
        b_regr = b_regr.view(batch, 1, 1, K, 1, 2)
        r_regr = _tranpose_and_gather_feat(r_regr, r_inds)
        r_regr = r_regr.view(batch, 1, 1, 1, K, 2)

        t_xs = t_xs + t_regr[..., 0]
        t_ys = t_ys + t_regr[..., 1]
        l_xs = l_xs + l_regr[..., 0]
        l_ys = l_ys + l_regr[..., 1]
        b_xs = b_xs + b_regr[..., 0]
        b_ys = b_ys + b_regr[..., 1]
        r_xs = r_xs + r_regr[..., 0]
        r_ys = r_ys + r_regr[..., 1]
    else:
        t_xs = t_xs + 0.5
        t_ys = t_ys + 0.5
        l_xs = l_xs + 0.5
        l_ys = l_ys + 0.5
        b_xs = b_xs + 0.5
        b_ys = b_ys + 0.5
        r_xs = r_xs + 0.5
        r_ys = r_ys + 0.5
      
    bboxes = torch.stack((l_xs, t_ys, r_xs, b_ys), dim=5)
    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)

    clses  = clses.contiguous().view(batch, -1, 1)
    clses  = _gather_feat(clses, inds).float()

    t_xs = t_xs.contiguous().view(batch, -1, 1)
    t_xs = _gather_feat(t_xs, inds).float()
    t_ys = t_ys.contiguous().view(batch, -1, 1)
    t_ys = _gather_feat(t_ys, inds).float()
    l_xs = l_xs.contiguous().view(batch, -1, 1)
    l_xs = _gather_feat(l_xs, inds).float()
    l_ys = l_ys.contiguous().view(batch, -1, 1)
    l_ys = _gather_feat(l_ys, inds).float()
    b_xs = b_xs.contiguous().view(batch, -1, 1)
    b_xs = _gather_feat(b_xs, inds).float()
    b_ys = b_ys.contiguous().view(batch, -1, 1)
    b_ys = _gather_feat(b_ys, inds).float()
    r_xs = r_xs.contiguous().view(batch, -1, 1)
    r_xs = _gather_feat(r_xs, inds).float()
    r_ys = r_ys.contiguous().view(batch, -1, 1)
    r_ys = _gather_feat(r_ys, inds).float()


    detections = torch.cat([bboxes, scores, t_xs, t_ys, l_xs, l_ys, 
                            b_xs, b_ys, r_xs, r_ys, clses], dim=2)

    return detections

def exct_decode(
    t_heat, l_heat, b_heat, r_heat, ct_heat, 
    t_regr=None, l_regr=None, b_regr=None, r_regr=None, 
    K=40, scores_thresh=0.1, center_thresh=0.1, aggr_weight=0.0, num_dets=1000
):
    batch, cat, height, width = t_heat.size()
    '''
    t_heat  = torch.sigmoid(t_heat)
    l_heat  = torch.sigmoid(l_heat)
    b_heat  = torch.sigmoid(b_heat)
    r_heat  = torch.sigmoid(r_heat)
    ct_heat = torch.sigmoid(ct_heat)
    '''

    if aggr_weight > 0:   
      t_heat = _h_aggregate(t_heat, aggr_weight=aggr_weight)
      l_heat = _v_aggregate(l_heat, aggr_weight=aggr_weight)
      b_heat = _h_aggregate(b_heat, aggr_weight=aggr_weight)
      r_heat = _v_aggregate(r_heat, aggr_weight=aggr_weight)
      
    # perform nms on heatmaps
    t_heat = _nms(t_heat)
    l_heat = _nms(l_heat)
    b_heat = _nms(b_heat)
    r_heat = _nms(r_heat)
      
    t_heat[t_heat > 1] = 1
    l_heat[l_heat > 1] = 1
    b_heat[b_heat > 1] = 1
    r_heat[r_heat > 1] = 1

    t_scores, t_inds, t_clses, t_ys, t_xs = _topk(t_heat, K=K)
    l_scores, l_inds, l_clses, l_ys, l_xs = _topk(l_heat, K=K)
    b_scores, b_inds, b_clses, b_ys, b_xs = _topk(b_heat, K=K)
    r_scores, r_inds, r_clses, r_ys, r_xs = _topk(r_heat, K=K)

    t_ys = t_ys.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    t_xs = t_xs.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    l_ys = l_ys.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    l_xs = l_xs.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    b_ys = b_ys.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    b_xs = b_xs.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    r_ys = r_ys.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
    r_xs = r_xs.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)

    t_clses = t_clses.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    l_clses = l_clses.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    b_clses = b_clses.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    r_clses = r_clses.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
    box_ct_xs = ((l_xs + r_xs + 0.5) / 2).long()
    box_ct_ys = ((t_ys + b_ys + 0.5) / 2).long()
    ct_inds = t_clses.long() * (height * width) + box_ct_ys * width + box_ct_xs
    ct_inds = ct_inds.view(batch, -1)
    ct_heat = ct_heat.view(batch, -1, 1)
    ct_scores = _gather_feat(ct_heat, ct_inds)

    t_scores = t_scores.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    l_scores = l_scores.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    b_scores = b_scores.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    r_scores = r_scores.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
    ct_scores = ct_scores.view(batch, K, K, K, K)
    scores    = (t_scores + l_scores + b_scores + r_scores + 2 * ct_scores) / 6

    # reject boxes based on classes
    cls_inds = (t_clses != l_clses) + (t_clses != b_clses) + \
               (t_clses != r_clses)
    cls_inds = (cls_inds > 0)

    top_inds  = (t_ys > l_ys) + (t_ys > b_ys) + (t_ys > r_ys)
    top_inds = (top_inds > 0)
    left_inds  = (l_xs > t_xs) + (l_xs > b_xs) + (l_xs > r_xs)
    left_inds = (left_inds > 0)
    bottom_inds  = (b_ys < t_ys) + (b_ys < l_ys) + (b_ys < r_ys)
    bottom_inds = (bottom_inds > 0)
    right_inds  = (r_xs < t_xs) + (r_xs < l_xs) + (r_xs < b_xs)
    right_inds = (right_inds > 0)

    sc_inds = (t_scores < scores_thresh) + (l_scores < scores_thresh) + \
              (b_scores < scores_thresh) + (r_scores < scores_thresh) + \
              (ct_scores < center_thresh)
    sc_inds = (sc_inds > 0)

    scores = scores - sc_inds.float()
    scores = scores - cls_inds.float()
    scores = scores - top_inds.float()
    scores = scores - left_inds.float()
    scores = scores - bottom_inds.float()
    scores = scores - right_inds.float()

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)

    if t_regr is not None and l_regr is not None \
      and b_regr is not None and r_regr is not None:
        t_regr = _tranpose_and_gather_feat(t_regr, t_inds)
        t_regr = t_regr.view(batch, K, 1, 1, 1, 2)
        l_regr = _tranpose_and_gather_feat(l_regr, l_inds)
        l_regr = l_regr.view(batch, 1, K, 1, 1, 2)
        b_regr = _tranpose_and_gather_feat(b_regr, b_inds)
        b_regr = b_regr.view(batch, 1, 1, K, 1, 2)
        r_regr = _tranpose_and_gather_feat(r_regr, r_inds)
        r_regr = r_regr.view(batch, 1, 1, 1, K, 2)

        t_xs = t_xs + t_regr[..., 0]
        t_ys = t_ys + t_regr[..., 1]
        l_xs = l_xs + l_regr[..., 0]
        l_ys = l_ys + l_regr[..., 1]
        b_xs = b_xs + b_regr[..., 0]
        b_ys = b_ys + b_regr[..., 1]
        r_xs = r_xs + r_regr[..., 0]
        r_ys = r_ys + r_regr[..., 1]
    else:
        t_xs = t_xs + 0.5
        t_ys = t_ys + 0.5
        l_xs = l_xs + 0.5
        l_ys = l_ys + 0.5
        b_xs = b_xs + 0.5
        b_ys = b_ys + 0.5
        r_xs = r_xs + 0.5
        r_ys = r_ys + 0.5
      
    bboxes = torch.stack((l_xs, t_ys, r_xs, b_ys), dim=5)
    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)

    clses  = t_clses.contiguous().view(batch, -1, 1)
    clses  = _gather_feat(clses, inds).float()

    t_xs = t_xs.contiguous().view(batch, -1, 1)
    t_xs = _gather_feat(t_xs, inds).float()
    t_ys = t_ys.contiguous().view(batch, -1, 1)
    t_ys = _gather_feat(t_ys, inds).float()
    l_xs = l_xs.contiguous().view(batch, -1, 1)
    l_xs = _gather_feat(l_xs, inds).float()
    l_ys = l_ys.contiguous().view(batch, -1, 1)
    l_ys = _gather_feat(l_ys, inds).float()
    b_xs = b_xs.contiguous().view(batch, -1, 1)
    b_xs = _gather_feat(b_xs, inds).float()
    b_ys = b_ys.contiguous().view(batch, -1, 1)
    b_ys = _gather_feat(b_ys, inds).float()
    r_xs = r_xs.contiguous().view(batch, -1, 1)
    r_xs = _gather_feat(r_xs, inds).float()
    r_ys = r_ys.contiguous().view(batch, -1, 1)
    r_ys = _gather_feat(r_ys, inds).float()


    detections = torch.cat([bboxes, scores, t_xs, t_ys, l_xs, l_ys, 
                            b_xs, b_ys, r_xs, r_ys, clses], dim=2)


    return detections

def ddd_decode(heat, rot, depth, dim, wh=None, reg=None, K=40):
    batch, cat, height, width = heat.size()
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
      
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
      reg = _tranpose_and_gather_feat(reg, inds)
      reg = reg.view(batch, K, 2)
      xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
      ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
      xs = xs.view(batch, K, 1) + 0.5
      ys = ys.view(batch, K, 1) + 0.5
      
    rot = _tranpose_and_gather_feat(rot, inds)
    rot = rot.view(batch, K, 8)
    depth = _tranpose_and_gather_feat(depth, inds)
    depth = depth.view(batch, K, 1)
    dim = _tranpose_and_gather_feat(dim, inds)
    dim = dim.view(batch, K, 3)
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    xs = xs.view(batch, K, 1)
    ys = ys.view(batch, K, 1)
      
    if wh is not None:
        wh = _tranpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, K, 2)
        detections = torch.cat(
            [xs, ys, scores, rot, depth, dim, wh, clses], dim=2)
    else:
        detections = torch.cat(
            [xs, ys, scores, rot, depth, dim, clses], dim=2)
      
    return detections

def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
      
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
      reg = _tranpose_and_gather_feat(reg, inds)
      reg = reg.view(batch, K, 2)
      xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
      ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
      xs = xs.view(batch, K, 1) + 0.5
      ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
      wh = wh.view(batch, K, cat, 2)
      clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
      wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
      wh = wh.view(batch, K, 2)
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
      
    return detections

def multi_pose_decode(
    heat, wh, kps, reg=None, hm_hp=None, hp_offset=None, K=100):
  batch, cat, height, width = heat.size()
  num_joints = kps.shape[1] // 2
  # heat = torch.sigmoid(heat)
  # perform nms on heatmaps
  heat = _nms(heat)
  scores, inds, clses, ys, xs = _topk(heat, K=K)

  kps = _tranpose_and_gather_feat(kps, inds)
  kps = kps.view(batch, K, num_joints * 2)
  kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
  kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
  if reg is not None:
    reg = _tranpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
  else:
    xs = xs.view(batch, K, 1) + 0.5
    ys = ys.view(batch, K, 1) + 0.5
  wh = _tranpose_and_gather_feat(wh, inds)
  wh = wh.view(batch, K, 2)
  clses  = clses.view(batch, K, 1).float()
  scores = scores.view(batch, K, 1)

  bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                      ys - wh[..., 1:2] / 2,
                      xs + wh[..., 0:1] / 2, 
                      ys + wh[..., 1:2] / 2], dim=2)
  if hm_hp is not None:
      hm_hp = _nms(hm_hp)
      thresh = 0.1
      kps = kps.view(batch, K, num_joints, 2).permute(
          0, 2, 1, 3).contiguous() # b x J x K x 2
      reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
      hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K) # b x J x K
      if hp_offset is not None:
          hp_offset = _tranpose_and_gather_feat(
              hp_offset, hm_inds.view(batch, -1))
          hp_offset = hp_offset.view(batch, num_joints, K, 2)
          hm_xs = hm_xs + hp_offset[:, :, :, 0]
          hm_ys = hm_ys + hp_offset[:, :, :, 1]
      else:
          hm_xs = hm_xs + 0.5
          hm_ys = hm_ys + 0.5
        
      mask = (hm_score > thresh).float()
      hm_score = (1 - mask) * -1 + mask * hm_score
      hm_ys = (1 - mask) * (-10000) + mask * hm_ys
      hm_xs = (1 - mask) * (-10000) + mask * hm_xs
      hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
          2).expand(batch, num_joints, K, K, 2)
      dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
      min_dist, min_ind = dist.min(dim=3) # b x J x K
      hm_score = hm_score.gather(2, min_ind).unsqueeze(-1) # b x J x K x 1
      min_dist = min_dist.unsqueeze(-1)
      min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
          batch, num_joints, K, 1, 2)
      hm_kps = hm_kps.gather(3, min_ind)
      hm_kps = hm_kps.view(batch, num_joints, K, 2)
      l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
             (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
             (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
      mask = (mask > 0).float().expand(batch, num_joints, K, 2)
      kps = (1 - mask) * hm_kps + mask * kps
      kps = kps.permute(0, 2, 1, 3).contiguous().view(
          batch, K, num_joints * 2)
  detections = torch.cat([bboxes, scores, kps, clses], dim=2)
    
  return detections