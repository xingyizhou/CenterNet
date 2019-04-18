cd src
# train
python main.py ctdet --exp_id coco_hg --arch hourglass --batch_size 24 --master_batch 4 --lr 2.5e-4 --load_model ../models/ExtremeNet_500000.pth --gpus 0,1,2,3,4
# test
python test.py ctdet --exp_id coco_hg --arch hourglass --keep_res --resume
# flip test
python test.py ctdet --exp_id coco_hg --arch hourglass --keep_res --resume --flip_test 
# multi scale test
python test.py ctdet --exp_id coco_hg --arch hourglass --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..