#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from dcn_v2 import DCNv2
from dcn_v2_func import DCNv2Function
from dcn_v2 import DCNv2Pooling
from dcn_v2_func import DCNv2PoolingFunction

deformable_groups = 1
N, inC, inH, inW = 2, 2, 4, 4
outC = 2
kH, kW = 3, 3

def conv_identify(weight, bias):
    weight.data.zero_()
    bias.data.zero_()
    o, i, h, w = weight.shape
    y = h//2
    x = w//2
    for p in range(i):
        for q in range(o):
            if p == q:
                weight.data[q, p, y, x] = 1.0

def check_zero_offset():
    conv_offset = nn.Conv2d(inC, deformable_groups * 2 * kH * kW,
        kernel_size=(kH, kW),
        stride=(1, 1),
        padding=(1, 1),
        bias=True).cuda()

    conv_mask = nn.Conv2d(inC, deformable_groups * 1 * kH * kW,
        kernel_size=(kH, kW),
        stride=(1, 1),
        padding=(1, 1),
        bias=True).cuda()

    dcn_v2 = DCNv2(inC, outC, (kH, kW),
                   stride=1, padding=1, dilation=1,
                   deformable_groups=deformable_groups).cuda()

    conv_offset.weight.data.zero_()
    conv_offset.bias.data.zero_()
    conv_mask.weight.data.zero_()
    conv_mask.bias.data.zero_()
    conv_identify(dcn_v2.weight, dcn_v2.bias)

    input = torch.randn(N, inC, inH, inW).cuda()
    offset = conv_offset(input)
    mask = conv_mask(input)
    mask = torch.sigmoid(mask)
    output = dcn_v2(input, offset, mask)
    output *= 2
    d = (input - output).abs().max()
    if d < 1e-10:
        print('Zero offset passed')
    else:
        print('Zero offset failed')

def check_gradient_dconv_double():

    input = torch.randn(N, inC, inH, inW, dtype=torch.float64).cuda()
    input.requires_grad = True

    offset = torch.randn(N, deformable_groups * 2 * kW * kH, inH, inW, dtype=torch.float64).cuda()
    # offset.data.zero_()
    # offset.data -= 0.00001
    offset.requires_grad = True

    mask = torch.rand(N, deformable_groups * 1 * kW * kH, inH, inW, dtype=torch.float64).cuda()
    # mask.data.zero_()
    mask.requires_grad = True
    mask = torch.sigmoid(mask)

    weight = torch.randn(outC, inC, kH, kW, dtype=torch.float64).cuda()
    weight.requires_grad = True

    bias = torch.rand(outC, dtype=torch.float64).cuda()
    bias.requires_grad = True

    func = DCNv2Function(stride=1, padding=1, dilation=1, deformable_groups=deformable_groups)

    print(gradcheck(func, (input, offset, mask, weight, bias), eps=1e-6, atol=1e-5, rtol=1e-3))

def check_gradient_dconv():

    input = torch.randn(N, inC, inH, inW).cuda()
    input.requires_grad = True

    offset = torch.randn(N, deformable_groups * 2 * kW * kH, inH, inW).cuda()
    # offset.data.zero_()
    # offset.data -= 0.5
    offset.requires_grad = True

    mask = torch.rand(N, deformable_groups * 1 * kW * kH, inH, inW).cuda()
    # mask.data.zero_()
    mask.requires_grad = True
    mask = torch.sigmoid(mask)

    weight = torch.randn(outC, inC, kH, kW).cuda()
    weight.requires_grad = True

    bias = torch.rand(outC).cuda()
    bias.requires_grad = True

    func = DCNv2Function(stride=1, padding=1, dilation=1, deformable_groups=deformable_groups)

    print(gradcheck(func, (input, offset, mask, weight, bias), eps=1e-3, atol=1e-3, rtol=1e-2))

def check_pooling_zero_offset():
    from dcn_v2 import DCNv2Pooling
    input = torch.randn(2, 16, 64, 64).cuda().zero_()
    input[0, :, 16:26, 16:26] = 1.
    input[1, :, 10:20, 20:30] = 2.
    rois = torch.tensor([
        [0, 65, 65, 103, 103],
        [1, 81, 41, 119, 79],
    ]).cuda().float()
    pooling = DCNv2Pooling(spatial_scale=1.0 / 4,
                           pooled_size=7,
                           output_dim=16,
                           no_trans=True,
                           group_size=1,
                           trans_std=0.1).cuda()

    out = pooling(input, rois, input.new())
    s = ', '.join(['%f' % out[i, :, :, :].mean().item() for i in range(rois.shape[0])])
    print(s)

    dpooling = DCNv2Pooling(spatial_scale=1.0 / 4,
                            pooled_size=7,
                            output_dim=16,
                            no_trans=False,
                            group_size=1,
                            trans_std=0.1).cuda()
    offset = torch.randn(20, 2, 7, 7).cuda().zero_()
    dout = dpooling(input, rois, offset)
    s = ', '.join(['%f' % dout[i, :, :, :].mean().item() for i in range(rois.shape[0])])
    print(s)

def check_gradient_dpooling():
    input = torch.randn(2, 3, 5, 5).cuda() * 0.01
    N = 4
    batch_inds = torch.randint(2, (N, 1)).cuda().float()
    x = torch.rand((N, 1)).cuda().float() * 15
    y = torch.rand((N, 1)).cuda().float() * 15
    w = torch.rand((N, 1)).cuda().float() * 10
    h = torch.rand((N, 1)).cuda().float() * 10
    rois = torch.cat((batch_inds, x, y, x + w, y + h), dim=1)
    offset = torch.randn(N, 2, 3, 3).cuda()
    dpooling = DCNv2Pooling(spatial_scale=1.0 / 4,
                            pooled_size=3,
                            output_dim=3,
                            no_trans=False,
                            group_size=1,
                            trans_std=0.0).cuda()
    input.requires_grad = True
    offset.requires_grad = True
    print('check_gradient_dpooling', gradcheck(dpooling, (input, rois, offset), eps=1e-4))


def example_dconv():
    from dcn_v2 import DCN
    input = torch.randn(2, 64, 128, 128).cuda()
    # wrap all things (offset and mask) in DCN
    dcn = DCN(64, 64, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2).cuda()
    output = dcn(input)
    targert = output.new(*output.size())
    targert.data.uniform_(-0.01, 0.01)
    error = (targert - output).mean()
    error.backward()
    print(output.shape)

def example_dpooling():
    from dcn_v2 import DCNv2Pooling
    input = torch.randn(2, 32, 64, 64).cuda()
    batch_inds = torch.randint(2, (20, 1)).cuda().float()
    x = torch.randint(256, (20, 1)).cuda().float()
    y = torch.randint(256, (20, 1)).cuda().float()
    w = torch.randint(64, (20, 1)).cuda().float()
    h = torch.randint(64, (20, 1)).cuda().float()
    rois = torch.cat((batch_inds, x, y, x + w, y + h), dim=1)
    offset = torch.randn(20, 2, 7, 7).cuda()
    input.requires_grad = True
    offset.requires_grad = True

    # normal roi_align
    pooling = DCNv2Pooling(spatial_scale=1.0 / 4,
                           pooled_size=7,
                           output_dim=32,
                           no_trans=True,
                           group_size=1,
                           trans_std=0.1).cuda()

    # deformable pooling
    dpooling = DCNv2Pooling(spatial_scale=1.0 / 4,
                            pooled_size=7,
                            output_dim=32,
                            no_trans=False,
                            group_size=1,
                            trans_std=0.1).cuda()

    out = pooling(input, rois, offset)
    dout = dpooling(input, rois, offset)
    print(out.shape)
    print(dout.shape)

    target_out = out.new(*out.size())
    target_out.data.uniform_(-0.01, 0.01)
    target_dout = dout.new(*dout.size())
    target_dout.data.uniform_(-0.01, 0.01)
    e = (target_out - out).mean()
    e.backward()
    e = (target_dout - dout).mean()
    e.backward()

def example_mdpooling():
    from dcn_v2 import DCNPooling
    input = torch.randn(2, 32, 64, 64).cuda()
    input.requires_grad = True
    batch_inds = torch.randint(2, (20, 1)).cuda().float()
    x = torch.randint(256, (20, 1)).cuda().float()
    y = torch.randint(256, (20, 1)).cuda().float()
    w = torch.randint(64, (20, 1)).cuda().float()
    h = torch.randint(64, (20, 1)).cuda().float()
    rois = torch.cat((batch_inds, x, y, x + w, y + h), dim=1)

    # mdformable pooling (V2)
    dpooling = DCNPooling(spatial_scale=1.0 / 4,
                         pooled_size=7,
                         output_dim=32,
                         no_trans=False,
                         group_size=1,
                         trans_std=0.1).cuda()

    dout = dpooling(input, rois)
    target = dout.new(*dout.size())
    target.data.uniform_(-0.1, 0.1)
    error = (target - dout).mean()
    error.backward()
    print(dout.shape)

if __name__ == '__main__':

    example_dconv()
    example_dpooling()
    example_mdpooling()

    check_pooling_zero_offset()
    # zero offset check
    if inC == outC:
        check_zero_offset()

    check_gradient_dpooling()

    # # gradient check
    # try:
    #     check_gradient_double()
    # except TypeError:
    #     print('''****** You can swith to double precision in dcn_v2_func.py by (un)commenting these two lines:
    #              ****** from _ext import dcn_v2 as _backend
    #              ****** from _ext import dcn_v2_double as _backend''')
    #     print('****** Your tensor may not be **double** type')
    #     print('****** Switching to **float** type')
    #
    #     check_gradient()
    # finally:
    #     print('****** Note: backward is not reentrant error may not be a serious problem, '
    #           '****** since the max error is less than 1e-7\n'
    #           '****** Still looking for what trigger this problem')