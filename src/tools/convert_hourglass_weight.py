from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

MODEL_PATH = '../../models/ExtremeNet_500000.pkl'
OUT_PATH = '../../models/ExtremeNet_500000.pth'

import torch
state_dict = torch.load(MODEL_PATH)
key_map = {'t_heats': 'hm_t', 'l_heats': 'hm_l', 'b_heats': 'hm_b', \
           'r_heats': 'hm_r', 'ct_heats': 'hm_c', \
           't_regrs': 'reg_t', 'l_regrs': 'reg_l', \
           'b_regrs': 'reg_b', 'r_regrs': 'reg_r'}

out = {}
for k in state_dict.keys():
  changed = False
  for m in key_map.keys():
    if m in k:
      if 'ct_heats' in k and m == 't_heats':
        continue
      new_k = k.replace(m, key_map[m])
      out[new_k] = state_dict[k]
      changed = True
      print('replace {} to {}'.format(k, new_k))
  if not changed:
    out[k] = state_dict[k]
data = {'epoch': 0,
        'state_dict': out}
torch.save(data, OUT_PATH)
