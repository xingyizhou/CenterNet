cd src
# train
python main.py ctdet --exp_id coco_res18_pretrained --arch res_18 --lr_step 45,60 --gpus 3
# test
python test.py ctdet --exp_id coco_res18_pretrained --arch res_18 --keep_res --resume
# flip test
python test.py ctdet --exp_id coco_res18_pretrained --arch res_18 --keep_res --resume --flip_test
# multi scale test
python test.py ctdet --exp_id coco_res18_pretrained --arch res_18 --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..
