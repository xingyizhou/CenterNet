# kitti_eval

`evaluate_object_3d_offline.cpp`evaluates your KITTI detection locally on your own computer using your validation data selected from KITTI training dataset, with the following metrics:

- overlap on image (AP)
- oriented overlap on image (AOS)
- overlap on ground-plane (AP)
- overlap in 3D (AP)

Compile `evaluate_object_3d_offline.cpp` with dependency of Boost and Linux `dirent.h` (You should already have it under most Linux).

Run the evalutaion by:

    ./evaluate_object_3d_offline groundtruth_dir result_dir
    
Note that you don't have to detect over all KITTI training data. The evaluator only evaluates samples whose result files exist.


### Updates

- June, 2017:
  * Fixed the bug of detection box filtering based on min height according to KITTI's note on 25.04.2017.
