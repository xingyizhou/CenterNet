cd src
# train
python main.py exdet --exp_id coco_dla --batch_size 64 --master_batch 1 --lr 2.5e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 8
# test
python test.py exdet --exp_id coco_dla --keep_res --resume
# flip test
python test.py exdet --exp_id coco_dla --keep_res --resume --flip_test 
# multi scale test
python test.py exdet --exp_id coco_dla --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..
