# MODEL ZOO

### Common settings and notes

- The experiments are run with pytorch 0.4.1, CUDA 9.0, and CUDNN 7.1.
- Training times are measured on our servers with 8 TITAN V GPUs (12 GB Memeory).
- Testing times are measured on our local machine with TITAN Xp GPU. 
- The models can be downloaded directly from [Google drive](https://drive.google.com/drive/folders/1S3NnppRgXea_IG4WeyquJcnOB3I6G-LX).

## Object Detection


### COCO

| Model                    | GPUs |Train time(h)| Test time (ms) |   AP               |  Download | 
|--------------------------|------|-------------|----------------|--------------------|-----------|
|[ctdet\_coco\_hg](../experiments/ctdet_coco_hg.sh)       |   5  |109          | 71 / 129 / 674 | 40.3 / 42.2 / 45.1 | [model](https://drive.google.com/file/d/13F904yXltGIqa_K3g3-FLleaNX0n7c9O) |
|[ctdet\_coco\_dla\_1x](../experiments/ctdet_coco_dla_1x.sh)  |   8  | 57          |  19 / 36 / 248 | 36.3 / 38.2 / 40.7 | [model](https://drive.google.com/file/d/1xqFsdA3DANMq1MfnXG8n7nQPe4AfXTHv) |
|[ctdet\_coco\_dla\_2x](../experiments/ctdet_coco_dla_2x.sh)  |   8  | 92          |  19 / 36 / 248 | 37.4 / 39.2 / 41.7 | [model](https://drive.google.com/file/d/18Q3fzzAsha_3Qid6mn4jcIFPeOGUaj1d) |
|[ctdet\_coco\_resdcn101](../experiments/ctdet_coco_resdcn101.sh)|   8  | 65          |  22 / 40 / 259 | 34.6 / 36.2 / 39.3 | [model](https://drive.google.com/file/d/1tKkSyzC3iWmM6XTYNJrC4XLCIToDmnHz) |
|[ctdet\_coco\_resdcn18](../experiments/ctdet_coco_resdcn18.sh) |   4  | 28          |  7 / 14 / 81   | 28.1 / 30.0 / 33.2 | [model](https://drive.google.com/file/d/1RtFps3kQAyLjQyzCao7pPDclOBQ64Vyp) |
|[exdet\_coco\_hg](../experiments/exdet_coco_hg.sh)       |   5  |215          | 134 / 246/1340 | 35.8 / 39.8 / 42.4 | [model](https://drive.google.com/file/d/1gf9bVWs9htzOaQiAF_mqaa2SCFTvwNvh) |
|[exdet\_coco\_dla](../experiments/exdet_coco_dla.sh)      |   8  |133          | 51 / 90 / 481  | 33.0 / 36.5 / 38.5 | [model](https://drive.google.com/file/d/1RtFps3kQAyLjQyzCao7pPDclOBQ64Vyp) |

#### Notes

- All models are trained on COCO train 2017 and evaluated on val 2017. 
- We show test time and AP with no augmentation / flip augmentation / multi scale (0.5, 0.75, 1, 1.25, 1.5) augmentation. 
- Results on COCO test-dev can be found in the paper or add `--trainval` for `test.py`. 
- exdet is our re-implementation of [ExtremeNet](https://github.com/xingyizhou/ExtremeNet). The testing does not include edge aggregation.
- For dla and resnets, `1x` means the training schedule that train 140 epochs with learning rate dropped 10 times at the 90 and 120 epoch (following [SimpleBaseline](https://github.com/Microsoft/human-pose-estimation.pytorch)). `2x` means train 230 epochs with learning rate dropped 10 times at the 180 and 210 epoch. The training schedules are **not** carefully investigated.
- The hourglass trained schedule follows [ExtremeNet](https://github.com/xingyizhou/ExtremeNet): trains 50 epochs (approximately 250000 iterations in batch size 24) and drops learning rate at the 40 epoch.
- Testing time include network forwarding time, decoding time, and nms time (for ExtremeNet).
- We observed up to 0.4 AP performance jitter due to randomness in training. 

### Pascal VOC

| Model                           |GPUs| Train time (h)| Test time (ms) | mAP  | Download  |
|---------------------------------|----|---------------|----------------|------|-----------|
|[ctdet\_pascal\_dla\_384](../experiments/ctdet_pascal_dla_384.sh)      | 1  |15             | 20             | 79.3 | [model](https://drive.google.com/file/d/19J7WoUZKYiSEU7vrhZInz5C2m0-9zmC1) |
|[ctdet\_pascal\_dla\_512](../experiments/ctdet_pascal_dla_512.sh)      | 2  |15             | 30             | 80.7 | [model](https://drive.google.com/file/d/1Iqqr_VLtG8T3tUzy-EuMaeuCtFuv6pEU) |
|[ctdet\_pascal\_resdcn18\_384](../experiments/ctdet_pascal_resdcn18_384.sh) | 1  |3              | 7              | 72.6 | [model](https://drive.google.com/file/d/1mpeslWUt0aJqx99_-UcCzJ7GBcD6sCA-) |
|[ctdet\_pascal\_resdcn18\_512](../experiments/ctdet_pascal_resdcn18_512.sh) | 1  |5              | 10             | 75.7 | [model](https://drive.google.com/file/d/1ILnmGrJQiDK2Ib-D7TzBs2fq5bfju-Bd) |
|[ctdet\_pascal\_resdcn101\_384](../experiments/ctdet_pascal_resdcn101_384.sh)| 2  |7              | 22             | 77.1 | [model](https://drive.google.com/file/d/1-Qy4zMhzmBk9fMF0u2s5zNI0Uj9latsk) |
|[ctdet\_pascal\_resdcn101\_512](../experiments/ctdet_pascal_resdcn101_512.sh)| 4  |7              | 33             | 78.7 | [model](https://drive.google.com/file/d/1TiJkPaofzUaEMLi5wr9TIIUY8G2VewCQ) |

#### Notes
- All models are trained on trainval 07+12 and tested on test 2007.
- Flip test is used by default.
- Training schedule: train for 70 epochs with learning rate dropped 10 times at the 45 and 60 epoch.
- We observed up to 1 mAP performance jitter due to randomness in training.

## Human pose estimation

### COCO

| Model                    | GPUs |Train time(h)| Test time (ms) |   AP        |  Download | 
|--------------------------|------|-------------|----------------|-------------|-----------|
|[multi\_pose\_hg_1x](../experiments/multi_pose_hg_1x.sh)    |   5  |62           | 151            | 58.7        | [model](https://drive.google.com/file/d/1neg1zfVjRTS45FCdk5hgIV-TxJ0McAew) |
|[multi\_pose\_hg_3x](../experiments/multi_pose_hg_3x.sh)    |   5  |188          | 151            | 64.0        | [model](https://drive.google.com/file/d/17-YS11iguOQKwPPHUg8LzakLzrgKSLfA) |
|[multi\_pose\_dla_1x](../experiments/multi_pose_dla_1x.sh)   |   8  |30           | 44             | 54.7        | [model](https://drive.google.com/file/d/1WKqVdkIew43SxLTSxi81a-_f4Q9v2Fx5) |
|[multi\_pose\_dla_3x](../experiments/multi_pose_dla_3x.sh)   |   8  |70           | 44             | 58.9        | [model](https://drive.google.com/file/d/1mC2PAQT_RuHi_9ZMZgkt4rg7BSY2_Lkd) |

#### Notes
- All models are trained on keypoint train 2017 images which contains at least one human with keypoint annotations (64115 images).
- The evaluation is done on COCO keypoint val 2017 (5000 images).
- Flip test is used by default.
- The models are fine-tuned from the corresponding center point detection models.
- Dla training schedule: `1x`: train for 140 epochs with learning rate dropped 10 times at the 90 and 120 epoch.`3x`: train for 320 epochs with learning rate dropped 10 times at the 270 and 300 epoch.
- Hourglass training schedule: `1x`: train for 50 epochs with learning rate dropped 10 times at the 40 epoch.`3x`: train for 150 epochs with learning rate dropped 10 times at the 130 epoch.

## 3D bounding box detection

#### Notes
- The 3dop split is from [3DOP](https://papers.nips.cc/paper/5644-3d-object-proposals-for-accurate-object-class-detection) and the suborn split is from [SubCNN](https://github.com/tanshen/SubCNN).
- No augmentation is used in testing.
- The models are trained for 70 epochs with learning rate dropped at the 45 and 60 epoch.

### KITTI 3DOP split

|Model       |GPUs|Train time|Test time|AP-E|AP-M|AP-H|AOS-E|AOS-M|AOS-H|BEV-E|BEV-M|BEV-H| Download |
|------------|----|----------|---------|----|----|----|-----|-----|-----|-----|-----|-----|----------|
|[ddd_3dop](../experiments/ddd_3dop.sh)|2   | 7h       |  31ms   |96.9|87.8|79.2|93.9 |84.3 |75.7 |34.0 |30.5 |26.8 | [model](https://drive.google.com/file/d/1LrAzVJqlZECVuyr_NJI_4xd88mA1fL5b)|

### KITTI SubCNN split

|Model       |GPUs|Train time|Test time|AP-E|AP-M|AP-H|AOS-E|AOS-M|AOS-H|BEV-E|BEV-M|BEV-H| Download |
|------------|----|----------|---------|----|----|----|-----|-----|-----|-----|-----|-----|----------|
|[ddd_sub](../experiments/ddd_sub.sh) |2   | 7h       |  31ms   |89.6|79.8|70.3|85.7 |75.2 |65.9 |34.9 |27.7 |26.4 | [model](https://drive.google.com/file/d/1ZUsUqrM_yTjxaJuDBxm0WTKXj1snVuvP)|