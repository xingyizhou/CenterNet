cd src
# train
python main.py ctdet --exp_id coco_resdcn101 --arch resdcn_101 --batch_size 96 --master_batch 5 --lr 3.75e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16
# test
python test.py ctdet --exp_id coco_resdcn101 --keep_res --resume
# flip test
python test.py ctdet --exp_id coco_resdcn101 --keep_res --resume --flip_test 
# multi scale test
python test.py ctdet --exp_id coco_resdcn101 --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..
