cd src
# train
python main.py ctdet --exp_id pascal_resdcn18_384 --arch resdcn_18 --dataset pascal --num_epochs 70 --lr_step 45,60
# test
python test.py ctdet --exp_id pascal_resdcn18_384 --arch resdcn_18 --dataset pascal --resume
# flip test
python test.py ctdet --exp_id pascal_resdcn18_384 --arch resdcn_18 --dataset pascal --resume --flip_test
cd ..
