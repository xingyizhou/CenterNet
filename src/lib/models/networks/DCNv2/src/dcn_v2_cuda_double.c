#include <THC/THC.h>
#include "cuda/dcn_v2_im2col_cuda_double.h"
#include "cuda/dcn_v2_psroi_pooling_cuda_double.h"

extern THCState *state;

// author: Charles Shang
// https://github.com/torch/cunn/blob/master/lib/THCUNN/generic/SpatialConvolutionMM.cu

void dcn_v2_cuda_forward(THCudaDoubleTensor *input, THCudaDoubleTensor *weight,
                         THCudaDoubleTensor *bias, THCudaDoubleTensor *ones,
                         THCudaDoubleTensor *offset, THCudaDoubleTensor *mask,
                         THCudaDoubleTensor *output, THCudaDoubleTensor *columns,
                         int kernel_h, int kernel_w,
                         const int stride_h, const int stride_w,
                         const int pad_h, const int pad_w,
                         const int dilation_h, const int dilation_w,
                         const int deformable_group)
{
    THCAssertSameGPU(THCudaDoubleTensor_checkGPU(state, 8, input, weight, bias, ones, offset, mask, output, columns));
    THArgCheck(THCudaDoubleTensor_isContiguous(state, input), 1, "input tensor has to be contiguous");
    THArgCheck(THCudaDoubleTensor_isContiguous(state, weight), 2, "weight tensor has to be contiguous");

    input = THCudaDoubleTensor_newContiguous(state, input);
    offset = THCudaDoubleTensor_newContiguous(state, offset);
    mask = THCudaDoubleTensor_newContiguous(state, mask);
    weight = THCudaDoubleTensor_newContiguous(state, weight);

    const int batch = THCudaDoubleTensor_size(state, input, 0);
    const int channels = THCudaDoubleTensor_size(state, input, 1);
    const int height = THCudaDoubleTensor_size(state, input, 2);
    const int width = THCudaDoubleTensor_size(state, input, 3);

    const int channels_out = THCudaDoubleTensor_size(state, weight, 0);
    const int channels_kernel = THCudaDoubleTensor_size(state, weight, 1);
    const int kernel_h_ = THCudaDoubleTensor_size(state, weight, 2);
    const int kernel_w_ = THCudaDoubleTensor_size(state, weight, 3);
    if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
        THError("Input shape and kernel shape wont match: (%d x %d vs %d x %d).",
                kernel_h_, kernel_w, kernel_h_, kernel_w_);
    if (channels != channels_kernel)
        THError("Input shape and kernel channels wont match: (%d vs %d).",
                channels, channels_kernel);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    if (THCudaDoubleTensor_nDimension(state, ones) != 2 ||
        THCudaDoubleTensor_size(state, ones, 0) * THCudaDoubleTensor_size(state, ones, 1) < height_out * width_out)
    {
        // Resize plane and fill with ones...
        THCudaDoubleTensor_resize2d(state, ones, height_out, width_out);
        THCudaDoubleTensor_fill(state, ones, 1);
    }

    // resize output
    THCudaDoubleTensor_resize4d(state, output, batch, channels_out, height_out, width_out);
    // resize temporary columns
    THCudaDoubleTensor_resize2d(state, columns, channels * kernel_h * kernel_w, 1 * height_out * width_out);

    THCudaDoubleTensor *input_n = THCudaDoubleTensor_new(state);
    THCudaDoubleTensor *offset_n = THCudaDoubleTensor_new(state);
    THCudaDoubleTensor *mask_n = THCudaDoubleTensor_new(state);
    THCudaDoubleTensor *output_n = THCudaDoubleTensor_new(state);

    for (int b = 0; b < batch; b++)
    {
        THCudaDoubleTensor_select(state, input_n, input, 0, b);
        THCudaDoubleTensor_select(state, offset_n, offset, 0, b);
        THCudaDoubleTensor_select(state, mask_n, mask, 0, b);
        THCudaDoubleTensor_select(state, output_n, output, 0, b);

        // Do Bias first:
        // M,N,K are dims of matrix A and B
        // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
        // (N x 1) (1 x M)
        long m_ = channels_out;
        long n_ = height_out * width_out;
        long k_ = 1;
        THCudaBlas_Dgemm(state, 't', 'n', n_, m_, k_, 1.0,
                         THCudaDoubleTensor_data(state, ones), k_,
                         THCudaDoubleTensor_data(state, bias), k_, 0.0,
                         THCudaDoubleTensor_data(state, output_n), n_);

        modulated_deformable_im2col_cuda(THCState_getCurrentStream(state),
                                         THCudaDoubleTensor_data(state, input_n), THCudaDoubleTensor_data(state, offset_n),
                                         THCudaDoubleTensor_data(state, mask_n),
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                                         deformable_group, THCudaDoubleTensor_data(state, columns));

        //(k * m)  x  (m * n)
        // Y = WC
        long m = channels_out;
        long n = height_out * width_out;
        long k = channels * kernel_h * kernel_w;
        THCudaBlas_Dgemm(state, 'n', 'n', n, m, k, 1.0f,
                         THCudaDoubleTensor_data(state, columns), n,
                         THCudaDoubleTensor_data(state, weight), k, 1.0f,
                         THCudaDoubleTensor_data(state, output_n), n);
    }
    THCudaDoubleTensor_free(state, input_n);
    THCudaDoubleTensor_free(state, offset_n);
    THCudaDoubleTensor_free(state, mask_n);
    THCudaDoubleTensor_free(state, output_n);

    THCudaDoubleTensor_free(state, input);
    THCudaDoubleTensor_free(state, offset);
    THCudaDoubleTensor_free(state, mask);
    THCudaDoubleTensor_free(state, weight);
}

void dcn_v2_cuda_backward(THCudaDoubleTensor *input, THCudaDoubleTensor *weight,
                          THCudaDoubleTensor *bias, THCudaDoubleTensor *ones,
                          THCudaDoubleTensor *offset, THCudaDoubleTensor *mask,
                          THCudaDoubleTensor *columns,
                          THCudaDoubleTensor *grad_input, THCudaDoubleTensor *grad_weight,
                          THCudaDoubleTensor *grad_bias, THCudaDoubleTensor *grad_offset,
                          THCudaDoubleTensor *grad_mask, THCudaDoubleTensor *grad_output,
                          int kernel_h, int kernel_w,
                          int stride_h, int stride_w,
                          int pad_h, int pad_w,
                          int dilation_h, int dilation_w,
                          int deformable_group)
{
    THCAssertSameGPU(THCudaDoubleTensor_checkGPU(state, 13, input, weight, bias, ones, offset, mask, columns,
                                                 grad_input, grad_weight, grad_bias, grad_offset, grad_mask, grad_output));
    THArgCheck(THCudaDoubleTensor_isContiguous(state, input), 1, "input tensor has to be contiguous");
    THArgCheck(THCudaDoubleTensor_isContiguous(state, weight), 2, "weight tensor has to be contiguous");

    input = THCudaDoubleTensor_newContiguous(state, input);
    offset = THCudaDoubleTensor_newContiguous(state, offset);
    mask = THCudaDoubleTensor_newContiguous(state, mask);
    weight = THCudaDoubleTensor_newContiguous(state, weight);
    grad_output = THCudaDoubleTensor_newContiguous(state, grad_output);

    const int batch = THCudaDoubleTensor_size(state, input, 0);
    const int channels = THCudaDoubleTensor_size(state, input, 1);
    const int height = THCudaDoubleTensor_size(state, input, 2);
    const int width = THCudaDoubleTensor_size(state, input, 3);

    const int channels_out = THCudaDoubleTensor_size(state, weight, 0);
    const int channels_kernel = THCudaDoubleTensor_size(state, weight, 1);
    const int kernel_h_ = THCudaDoubleTensor_size(state, weight, 2);
    const int kernel_w_ = THCudaDoubleTensor_size(state, weight, 3);
    if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
        THError("Input shape and kernel shape wont match: (%d x %d vs %d x %d).",
                kernel_h_, kernel_w, kernel_h_, kernel_w_);
    if (channels != channels_kernel)
        THError("Input shape and kernel channels wont match: (%d vs %d).",
                channels, channels_kernel);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    if (THCudaDoubleTensor_nDimension(state, ones) != 2 ||
        THCudaDoubleTensor_size(state, ones, 0) * THCudaDoubleTensor_size(state, ones, 1) < height_out * width_out)
    {
        // Resize plane and fill with ones...
        THCudaDoubleTensor_resize2d(state, ones, height_out, width_out);
        THCudaDoubleTensor_fill(state, ones, 1);
    }

    // THCudaDoubleTensor_resize4d(state, grad_input, batch, channels, height, width);
    THCudaDoubleTensor_resize2d(state, columns, channels * kernel_h * kernel_w, height_out * width_out);

    THCudaDoubleTensor *input_n = THCudaDoubleTensor_new(state);
    THCudaDoubleTensor *offset_n = THCudaDoubleTensor_new(state);
    THCudaDoubleTensor *mask_n = THCudaDoubleTensor_new(state);

    THCudaDoubleTensor *grad_output_n = THCudaDoubleTensor_new(state);
    THCudaDoubleTensor *grad_input_n = THCudaDoubleTensor_new(state);
    THCudaDoubleTensor *grad_offset_n = THCudaDoubleTensor_new(state);
    THCudaDoubleTensor *grad_mask_n = THCudaDoubleTensor_new(state);

    for (int b = 0; b < batch; b++)
    {
        THCudaDoubleTensor_select(state, input_n, input, 0, b);
        THCudaDoubleTensor_select(state, offset_n, offset, 0, b);
        THCudaDoubleTensor_select(state, mask_n, mask, 0, b);
        THCudaDoubleTensor_select(state, grad_output_n, grad_output, 0, b);
        THCudaDoubleTensor_select(state, grad_input_n, grad_input, 0, b);
        THCudaDoubleTensor_select(state, grad_offset_n, grad_offset, 0, b);
        THCudaDoubleTensor_select(state, grad_mask_n, grad_mask, 0, b);

        long m = channels * kernel_h * kernel_w;
        long n = height_out * width_out;
        long k = channels_out;

        THCudaBlas_Dgemm(state, 'n', 't', n, m, k, 1.0,
                         THCudaDoubleTensor_data(state, grad_output_n), n,
                         THCudaDoubleTensor_data(state, weight), m, 0.0,
                         THCudaDoubleTensor_data(state, columns), n);

        // gradient w.r.t. input offset and mask data
        modulated_deformable_col2im_coord_cuda(THCState_getCurrentStream(state),
                                               THCudaDoubleTensor_data(state, columns),
                                               THCudaDoubleTensor_data(state, input_n),
                                               THCudaDoubleTensor_data(state, offset_n),
                                               THCudaDoubleTensor_data(state, mask_n),
                                               1, channels, height, width,
                                               height_out, width_out, kernel_h, kernel_w,
                                               pad_h, pad_w, stride_h, stride_w,
                                               dilation_h, dilation_w, deformable_group,
                                               THCudaDoubleTensor_data(state, grad_offset_n),
                                               THCudaDoubleTensor_data(state, grad_mask_n));
        // gradient w.r.t. input data
        modulated_deformable_col2im_cuda(THCState_getCurrentStream(state),
                                         THCudaDoubleTensor_data(state, columns),
                                         THCudaDoubleTensor_data(state, offset_n),
                                         THCudaDoubleTensor_data(state, mask_n),
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         THCudaDoubleTensor_data(state, grad_input_n));

        // gradient w.r.t. weight, dWeight should accumulate across the batch and group
        modulated_deformable_im2col_cuda(THCState_getCurrentStream(state),
                                         THCudaDoubleTensor_data(state, input_n),
                                         THCudaDoubleTensor_data(state, offset_n),
                                         THCudaDoubleTensor_data(state, mask_n),
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         THCudaDoubleTensor_data(state, columns));
        long m_ = channels_out;
        long n_ = channels * kernel_h * kernel_w;
        long k_ = height_out * width_out;

        THCudaBlas_Dgemm(state, 't', 'n', n_, m_, k_, 1.0,
                         THCudaDoubleTensor_data(state, columns), k_,
                         THCudaDoubleTensor_data(state, grad_output_n), k_, 1.0,
                         THCudaDoubleTensor_data(state, grad_weight), n_);

        // gradient w.r.t. bias
        // long m_ = channels_out;
        // long k__ = height_out * width_out;
        THCudaBlas_Dgemv(state,
                         't',
                         k_, m_, 1.0,
                         THCudaDoubleTensor_data(state, grad_output_n), k_,
                         THCudaDoubleTensor_data(state, ones), 1, 1.0,
                         THCudaDoubleTensor_data(state, grad_bias), 1);
    }

    THCudaDoubleTensor_free(state, input_n);
    THCudaDoubleTensor_free(state, offset_n);
    THCudaDoubleTensor_free(state, mask_n);

    THCudaDoubleTensor_free(state, grad_output_n);
    THCudaDoubleTensor_free(state, grad_input_n);
    THCudaDoubleTensor_free(state, grad_offset_n);
    THCudaDoubleTensor_free(state, grad_mask_n);

    THCudaDoubleTensor_free(state, input);
    THCudaDoubleTensor_free(state, offset);
    THCudaDoubleTensor_free(state, mask);
    THCudaDoubleTensor_free(state, weight);
    THCudaDoubleTensor_free(state, grad_output);
}


void dcn_v2_psroi_pooling_cuda_forward(THCudaDoubleTensor * input, THCudaDoubleTensor * bbox,
                                       THCudaDoubleTensor * trans, 
                                       THCudaDoubleTensor * out, THCudaDoubleTensor * top_count,
                                       const int no_trans,
                                       const double spatial_scale,
                                       const int output_dim,
                                       const int group_size,
                                       const int pooled_size,
                                       const int part_size,
                                       const int sample_per_part,
                                       const double trans_std)
{
    THArgCheck(THCudaDoubleTensor_isContiguous(state, input), 1, "input tensor has to be contiguous");
    THCAssertSameGPU(THCudaDoubleTensor_checkGPU(state, 5, input, bbox, trans, out, top_count));

    const int batch = THCudaDoubleTensor_size(state, input, 0);
    const int channels = THCudaDoubleTensor_size(state, input, 1);
    const int height = THCudaDoubleTensor_size(state, input, 2);
    const int width = THCudaDoubleTensor_size(state, input, 3);
    const int channels_trans = no_trans? 2 : THCudaDoubleTensor_size(state, trans, 1);

    const int num_bbox = THCudaDoubleTensor_size(state, bbox, 0);
    if (num_bbox != THCudaDoubleTensor_size(state, out, 0))
        THError("Output shape and bbox number wont match: (%d vs %d).", 
                THCudaDoubleTensor_size(state, out, 0), num_bbox);

    DeformablePSROIPoolForward(THCState_getCurrentStream(state),
                               THCudaDoubleTensor_data(state, input),
                               THCudaDoubleTensor_data(state, bbox),
                               THCudaDoubleTensor_data(state, trans),
                               THCudaDoubleTensor_data(state, out),
                               THCudaDoubleTensor_data(state, top_count),
                               batch, channels, height, width,
                               num_bbox, 
                               channels_trans, 
                               no_trans, 
                               spatial_scale,
                               output_dim, 
                               group_size, 
                               pooled_size, 
                               part_size,
                               sample_per_part, 
                               trans_std);
}

void dcn_v2_psroi_pooling_cuda_backward(THCudaDoubleTensor * out_grad, 
                                        THCudaDoubleTensor * input, THCudaDoubleTensor * bbox,
                                        THCudaDoubleTensor * trans, THCudaDoubleTensor * top_count,
                                        THCudaDoubleTensor * input_grad, THCudaDoubleTensor * trans_grad,
                                        const int no_trans,
                                        const double spatial_scale,
                                        const int output_dim,
                                        const int group_size,
                                        const int pooled_size,
                                        const int part_size,
                                        const int sample_per_part,
                                        const double trans_std)
{
    THArgCheck(THCudaDoubleTensor_isContiguous(state, out_grad), 0, "out_grad tensor has to be contiguous");
    THArgCheck(THCudaDoubleTensor_isContiguous(state, input), 1, "input tensor has to be contiguous");
    THCAssertSameGPU(THCudaDoubleTensor_checkGPU(state, 7, input, bbox, trans, out_grad, top_count,
                    input_grad, trans_grad));

    const int batch = THCudaDoubleTensor_size(state, input, 0);
    const int channels = THCudaDoubleTensor_size(state, input, 1);
    const int height = THCudaDoubleTensor_size(state, input, 2);
    const int width = THCudaDoubleTensor_size(state, input, 3);
    const int channels_trans = no_trans? 2 : THCudaDoubleTensor_size(state, trans, 1);

    const int num_bbox = THCudaDoubleTensor_size(state, bbox, 0);
    if (num_bbox != THCudaDoubleTensor_size(state, out_grad, 0))
        THError("Output shape and bbox number wont match: (%d vs %d).", 
                THCudaDoubleTensor_size(state, out_grad, 0), num_bbox);

    DeformablePSROIPoolBackwardAcc(THCState_getCurrentStream(state),
                                   THCudaDoubleTensor_data(state, out_grad),
                                   THCudaDoubleTensor_data(state, input),
                                   THCudaDoubleTensor_data(state, bbox),
                                   THCudaDoubleTensor_data(state, trans),
                                   THCudaDoubleTensor_data(state, top_count),
                                   THCudaDoubleTensor_data(state, input_grad),
                                   THCudaDoubleTensor_data(state, trans_grad),
                                   batch, channels, height, width, num_bbox,
                                   channels_trans, 
                                   no_trans, 
                                   spatial_scale, 
                                   output_dim,
                                   group_size, 
                                   pooled_size, 
                                   part_size,
                                   sample_per_part, 
                                   trans_std);
}