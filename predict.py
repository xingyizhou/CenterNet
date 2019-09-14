from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import cv2
from detectors.ctdet import CtdetDetector


image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

color_list = np.array([1.000, 1.000, 1.000, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494, 0.184, 0.556,
                       0.466, 0.674, 0.188, 0.301, 0.745, 0.933, 0.635, 0.078, 0.184, 0.300, 0.300, 0.300,
                       0.600, 0.600, 0.600, 1.000, 0.000, 0.000, 1.000, 0.500, 0.000, 0.749, 0.749, 0.000,
                       0.000, 1.000, 0.000, 0.000, 0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.667, 0.000,
                       0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000, 0.667, 1.000, 0.000,
                       1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000, 1.000, 0.000, 0.000, 0.333, 0.500,
                       0.000, 0.667, 0.500, 0.000, 1.000, 0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500,
                       0.333, 0.667, 0.500, 0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500,
                       0.667, 0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333, 0.500,
                       1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000, 0.000, 0.667, 1.000,
                       0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333, 0.333, 1.000, 0.333, 0.667, 1.000,
                       0.333, 1.000, 1.000, 0.667, 0.000, 1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000,
                       0.667, 1.000, 1.000, 1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000,
                       0.167, 0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000,
                       0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000,
                       0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000,
                       0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667,
                       0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143,
                       0.286, 0.286, 0.286, 0.429, 0.429, 0.429, 0.571, 0.571, 0.571, 0.714, 0.714, 0.714,
                       0.857, 0.857, 0.857, 0.000, 0.447, 0.741, 0.50, 0.5, 0]).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255

colors = [(color_list[_]).astype(np.uint8) for _ in range(len(color_list))]
colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)


class DotDict(dict):
    def __getattr__(self, item):
        return self[item]


def add_coco_bbox(img, bbox, cat, conf=1, show_txt=True):
    # bbox = np.array(bbox, dtype=np.int32)
    # cat = (int(cat) + 1) % 80
    cat = int(cat)
    # print('cat', cat, self.names[cat])
    c = colors[cat][0][0].tolist()
    txt = '{}{:.1f}'.format(['ov', 'mif'][cat], conf)  # my own class
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
    if show_txt:
        cv2.rectangle(img,
                      (bbox[0], bbox[1] - cat_size[1] - 2),
                      (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
        cv2.putText(img, txt, (bbox[0], bbox[1] - 2),
                    font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    return img


def add_2d_detection(img, dets, show_txt=True, center_thresh=0.5):
    for cat in dets:
        for i in range(len(dets[cat])):
            if dets[cat][i, -1] > center_thresh:
                bbox = dets[cat][i, :4]
                add_coco_bbox(img,
                              [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                              cat - 1,
                              dets[cat][i, -1],
                              show_txt=show_txt)
    return img


if __name__ == '__main__':
    opt = DotDict({'demo': './images/val',
                   'gpus': [0],
                   'device': None,
                   'arch': 'res_50',
                   'heads': {'hm': 2, 'wh': 2, 'reg': 2},
                   'head_conv': 64,
                   'load_model': 'model_best.pth',
                   'test_scales': [1.0],
                   'mean': [0.408, 0.447, 0.470],
                   'std': [0.289, 0.274, 0.278],
                   'debugger_theme': 'dark',
                   'debug': 0,
                   'down_ratio': 4,
                   'pad': 31,
                   'fix_res': True,
                   'input_h': 800,
                   'input_w': 800,
                   'flip': 0.5,
                   'flip_test': False,
                   'nms': False,
                   'num_classes': 2,
                   'reg_offset': True,
                   'K': 100,
                   })
    detector = CtdetDetector(opt)

    if os.path.isdir(opt.demo):
        image_names = []
        ls = os.listdir(opt.demo)
        for file_name in sorted(ls):
            ext = file_name[file_name.rfind('.') + 1:].lower()
            if ext in image_ext:
                image_names.append(os.path.join(opt.demo, file_name))
    else:
        image_names = [opt.demo]

    for (image_name) in image_names:
        ret = detector.run(image_name)

        img = cv2.imread(image_name)
        img = add_2d_detection(img, ret['results'])
        cv2.imwrite('./prd/'+os.path.basename(image_name), img)

        time_str = ''
        for stat in time_stats:
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)
