import os
import torch
from torch.utils.ffi import create_extension


sources = ['src/dcn_v2_double.c']
headers = ['src/dcn_v2_double.h']
defines = []
with_cuda = False

extra_objects = []
if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/dcn_v2_cuda_double.c']
    headers += ['src/dcn_v2_cuda_double.h']
    defines += [('WITH_CUDA', None)]
    extra_objects += ['src/cuda/dcn_v2_im2col_cuda_double.cu.o']
    extra_objects += ['src/cuda/dcn_v2_psroi_pooling_cuda_double.cu.o']
    with_cuda = True
else:
    raise ValueError('CUDA is not available')

extra_compile_args = ['-fopenmp', '-std=c99']

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
sources = [os.path.join(this_file, fname) for fname in sources]
headers = [os.path.join(this_file, fname) for fname in headers]
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    '_ext.dcn_v2_double',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects,
    extra_compile_args=extra_compile_args
)

if __name__ == '__main__':
    ffi.build()
