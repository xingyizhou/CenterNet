#!/usr/bin/env bash
cd src/cuda

# compile dcn
nvcc -c -o dcn_v2_im2col_cuda.cu.o dcn_v2_im2col_cuda.cu -x cu -Xcompiler -fPIC
nvcc -c -o dcn_v2_im2col_cuda_double.cu.o dcn_v2_im2col_cuda_double.cu -x cu -Xcompiler -fPIC

# compile dcn-roi-pooling
nvcc -c -o dcn_v2_psroi_pooling_cuda.cu.o dcn_v2_psroi_pooling_cuda.cu -x cu -Xcompiler -fPIC
nvcc -c -o dcn_v2_psroi_pooling_cuda_double.cu.o dcn_v2_psroi_pooling_cuda_double.cu -x cu -Xcompiler -fPIC

cd -
python build.py
python build_double.py
