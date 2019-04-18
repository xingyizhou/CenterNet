cd src
# train
python main.py ctdet --exp_id pascal_dla_512 --dataset pascal --input_res 512 --num_epochs 70 --lr_step 45,60 --gpus 0,1
# test
python test.py ctdet --exp_id pascal_dla_512 --dataset pascal --input_res 512 --resume
# flip test
python test.py ctdet --exp_id pascal_dla_512 --dataset pascal --input_res 512 --resume --flip_test
cd ..
