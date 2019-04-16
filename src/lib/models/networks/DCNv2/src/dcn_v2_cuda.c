#include <THC/THC.h>
#include "cuda/dcn_v2_im2col_cuda.h"
#include "cuda/dcn_v2_psroi_pooling_cuda.h"

extern THCState *state;

// author: Charles Shang
// https://github.com/torch/cunn/blob/master/lib/THCUNN/generic/SpatialConvolutionMM.cu

void dcn_v2_cuda_forward(THCudaTensor *input, THCudaTensor *weight,
                         THCudaTensor *bias, THCudaTensor *ones,
                         THCudaTensor *offset, THCudaTensor *mask,
                         THCudaTensor *output, THCudaTensor *columns,
                         int kernel_h, int kernel_w,
                         const int stride_h, const int stride_w,
                         const int pad_h, const int pad_w,
                         const int dilation_h, const int dilation_w,
                         const int deformable_group)
{
    THCAssertSameGPU(THCudaTensor_checkGPU(state, 8, input, weight, bias, ones, offset, mask, output, columns));
    THArgCheck(THCudaTensor_isContiguous(state, input), 1, "input tensor has to be contiguous");
    THArgCheck(THCudaTensor_isContiguous(state, weight), 2, "weight tensor has to be contiguous");
    
    const int batch = THCudaTensor_size(state, input, 0);
    const int channels = THCudaTensor_size(state, input, 1);
    const int height = THCudaTensor_size(state, input, 2);
    const int width = THCudaTensor_size(state, input, 3);

    const int channels_out = THCudaTensor_size(state, weight, 0);
    const int channels_kernel = THCudaTensor_size(state, weight, 1);
    const int kernel_h_ = THCudaTensor_size(state, weight, 2);
    const int kernel_w_ = THCudaTensor_size(state, weight, 3);
    if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
        THError("Input shape and kernel shape wont match: (%d x %d vs %d x %d).", 
        kernel_h_, kernel_w, kernel_h_, kernel_w_);
    if (channels != channels_kernel)
        THError("Input shape and kernel channels wont match: (%d vs %d).", 
        channels, channels_kernel);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    if (THCudaTensor_nDimension(state, ones) != 2 ||
        THCudaTensor_size(state, ones, 0) * THCudaTensor_size(state, ones, 1) < height_out * width_out)
    {
        // Resize plane and fill with ones...
        THCudaTensor_resize2d(state, ones, height_out, width_out);
        THCudaTensor_fill(state, ones, 1);
    }

    // resize output
    THCudaTensor_resize4d(state, output, batch, channels_out, height_out, width_out);
    // resize temporary columns
    THCudaTensor_resize2d(state, columns, channels * kernel_h * kernel_w, 1 * height_out * width_out);

    THCudaTensor *input_n = THCudaTensor_new(state);
    THCudaTensor *offset_n = THCudaTensor_new(state);
    THCudaTensor *mask_n = THCudaTensor_new(state);
    THCudaTensor *output_n = THCudaTensor_new(state);

    for (int b = 0; b < batch; b++)
    {
        THCudaTensor_select(state, input_n, input, 0, b);
        THCudaTensor_select(state, offset_n, offset, 0, b);
        THCudaTensor_select(state, mask_n, mask, 0, b);
        THCudaTensor_select(state, output_n, output, 0, b);

        // Do Bias first:
        // M,N,K are dims of matrix A and B
        // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
        // (N x 1) (1 x M)
        long m_ = channels_out;
        long n_ = height_out * width_out;
        long k_ = 1;
        THCudaBlas_Sgemm(state, 't', 'n', n_, m_, k_, 1.0f,
                         THCudaTensor_data(state, ones), k_,
                         THCudaTensor_data(state, bias), k_, 0.0f,
                         THCudaTensor_data(state, output_n), n_);

        modulated_deformable_im2col_cuda(THCState_getCurrentStream(state),
                                         THCudaTensor_data(state, input_n), THCudaTensor_data(state, offset_n),
                                         THCudaTensor_data(state, mask_n),
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                                         deformable_group, THCudaTensor_data(state, columns));

        //(k * m)  x  (m * n)
        // Y = WC
        long m = channels_out;
        long n = height_out * width_out;
        long k = channels * kernel_h * kernel_w;
        THCudaBlas_Sgemm(state, 'n', 'n', n, m, k, 1.0f,
                         THCudaTensor_data(state, columns), n,
                         THCudaTensor_data(state, weight), k, 1.0f,
                         THCudaTensor_data(state, output_n), n);
    }
    THCudaTensor_free(state, input_n);
    THCudaTensor_free(state, offset_n);
    THCudaTensor_free(state, mask_n);
    THCudaTensor_free(state, output_n);
}

void dcn_v2_cuda_backward(THCudaTensor *input, THCudaTensor *weight,
                          THCudaTensor *bias, THCudaTensor *ones,
                          THCudaTensor *offset, THCudaTensor *mask,
                          THCudaTensor *columns,
                          THCudaTensor *grad_input, THCudaTensor *grad_weight,
                          THCudaTensor *grad_bias, THCudaTensor *grad_offset,
                          THCudaTensor *grad_mask, THCudaTensor *grad_output,
                          int kernel_h, int kernel_w,
                          int stride_h, int stride_w,
                          int pad_h, int pad_w,
                          int dilation_h, int dilation_w,
                          int deformable_group)
{
    THCAssertSameGPU(THCudaTensor_checkGPU(state, 13, input, weight, bias, ones, offset, mask, columns,
                                           grad_input, grad_weight, grad_bias, grad_offset, grad_mask, grad_output));
    THArgCheck(THCudaTensor_isContiguous(state, input), 1, "input tensor has to be contiguous");
    THArgCheck(THCudaTensor_isContiguous(state, weight), 2, "weight tensor has to be contiguous");

    const int batch = THCudaTensor_size(state, input, 0);
    const int channels = THCudaTensor_size(state, input, 1);
    const int height = THCudaTensor_size(state, input, 2);
    const int width = THCudaTensor_size(state, input, 3);

    const int channels_out = THCudaTensor_size(state, weight, 0);
    const int channels_kernel = THCudaTensor_size(state, weight, 1);
    const int kernel_h_ = THCudaTensor_size(state, weight, 2);
    const int kernel_w_ = THCudaTensor_size(state, weight, 3);
    if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
        THError("Input shape and kernel shape wont match: (%d x %d vs %d x %d).", 
        kernel_h_, kernel_w, kernel_h_, kernel_w_);
    if (channels != channels_kernel)
        THError("Input shape and kernel channels wont match: (%d vs %d).", 
        channels, channels_kernel);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    if (THCudaTensor_nDimension(state, ones) != 2 ||
        THCudaTensor_size(state, ones, 0) * THCudaTensor_size(state, ones, 1) < height_out * width_out)
    {
        // Resize plane and fill with ones...
        THCudaTensor_resize2d(state, ones, height_out, width_out);
        THCudaTensor_fill(state, ones, 1.0f);
    }

    THCudaTensor_resize4d(state, grad_input, batch, channels, height, width);
    THCudaTensor_resize2d(state, columns, channels * kernel_h * kernel_w, height_out * width_out);

    THCudaTensor *input_n = THCudaTensor_new(state);
    THCudaTensor *offset_n = THCudaTensor_new(state);
    THCudaTensor *mask_n = THCudaTensor_new(state);

    THCudaTensor *grad_output_n = THCudaTensor_new(state);
    THCudaTensor *grad_input_n = THCudaTensor_new(state);
    THCudaTensor *grad_offset_n = THCudaTensor_new(state);
    THCudaTensor *grad_mask_n = THCudaTensor_new(state);

    for (int b = 0; b < batch; b++)
    {
        THCudaTensor_select(state, input_n, input, 0, b);
        THCudaTensor_select(state, offset_n, offset, 0, b);
        THCudaTensor_select(state, mask_n, mask, 0, b);
        THCudaTensor_select(state, grad_output_n, grad_output, 0, b);
        THCudaTensor_select(state, grad_input_n, grad_input, 0, b);
        THCudaTensor_select(state, grad_offset_n, grad_offset, 0, b);
        THCudaTensor_select(state, grad_mask_n, grad_mask, 0, b);

        long m = channels * kernel_h * kernel_w;
        long n = height_out * width_out;
        long k = channels_out;

        THCudaBlas_Sgemm(state, 'n', 't', n, m, k, 1.0f,
                         THCudaTensor_data(state, grad_output_n), n,
                         THCudaTensor_data(state, weight), m, 0.0f,
                         THCudaTensor_data(state, columns), n);

        // gradient w.r.t. input coordinate data
        modulated_deformable_col2im_coord_cuda(THCState_getCurrentStream(state),
                                               THCudaTensor_data(state, columns),
                                               THCudaTensor_data(state, input_n),
                                               THCudaTensor_data(state, offset_n),
                                               THCudaTensor_data(state, mask_n),
                                               1, channels, height, width,
                                               height_out, width_out, kernel_h, kernel_w,
                                               pad_h, pad_w, stride_h, stride_w,
                                               dilation_h, dilation_w, deformable_group,
                                               THCudaTensor_data(state, grad_offset_n),
                                               THCudaTensor_data(state, grad_mask_n));
        // gradient w.r.t. input data
        modulated_deformable_col2im_cuda(THCState_getCurrentStream(state),
                                         THCudaTensor_data(state, columns),
                                         THCudaTensor_data(state, offset_n),
                                         THCudaTensor_data(state, mask_n),
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         THCudaTensor_data(state, grad_input_n));

        // gradient w.r.t. weight, dWeight should accumulate across the batch and group
        modulated_deformable_im2col_cuda(THCState_getCurrentStream(state),
                                         THCudaTensor_data(state, input_n),
                                         THCudaTensor_data(state, offset_n),
                                         THCudaTensor_data(state, mask_n),
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         THCudaTensor_data(state, columns));
        long m_ = channels_out;
        long n_ = channels * kernel_h * kernel_w;
        long k_ = height_out * width_out;

        THCudaBlas_Sgemm(state, 't', 'n', n_, m_, k_, 1.0f,
                         THCudaTensor_data(state, columns), k_,
                         THCudaTensor_data(state, grad_output_n), k_, 1.0f,
                         THCudaTensor_data(state, grad_weight), n_);

        // gradient w.r.t. bias
        // long m_ = channels_out;
        // long k__ = height_out * width_out;
        THCudaBlas_Sgemv(state,
                         't',
                         k_, m_, 1.0f,
                         THCudaTensor_data(state, grad_output_n), k_,
                         THCudaTensor_data(state, ones), 1, 1.0f,
                         THCudaTensor_data(state, grad_bias), 1);
    }

    THCudaTensor_free(state, input_n);
    THCudaTensor_free(state, offset_n);
    THCudaTensor_free(state, mask_n);

    THCudaTensor_free(state, grad_output_n);
    THCudaTensor_free(state, grad_input_n);
    THCudaTensor_free(state, grad_offset_n);
    THCudaTensor_free(state, grad_mask_n);
}

void dcn_v2_psroi_pooling_cuda_forward(THCudaTensor * input, THCudaTensor * bbox,
                                       THCudaTensor * trans, 
                                       THCudaTensor * out, THCudaTensor * top_count,
                                       const int no_trans,
                                       const float spatial_scale,
                                       const int output_dim,
                                       const int group_size,
                                       const int pooled_size,
                                       const int part_size,
                                       const int sample_per_part,
                                       const float trans_std)
{
    THArgCheck(THCudaTensor_isContiguous(state, input), 1, "input tensor has to be contiguous");
    THCAssertSameGPU(THCudaTensor_checkGPU(state, 5, input, bbox, trans, out, top_count));

    const int batch = THCudaTensor_size(state, input, 0);
    const int channels = THCudaTensor_size(state, input, 1);
    const int height = THCudaTensor_size(state, input, 2);
    const int width = THCudaTensor_size(state, input, 3);
    const int channels_trans = no_trans? 2 : THCudaTensor_size(state, trans, 1);

    const int num_bbox = THCudaTensor_size(state, bbox, 0);
    if (num_bbox != THCudaTensor_size(state, out, 0))
        THError("Output shape and bbox number wont match: (%d vs %d).", 
                THCudaTensor_size(state, out, 0), num_bbox);

    DeformablePSROIPoolForward(THCState_getCurrentStream(state),
                               THCudaTensor_data(state, input),
                               THCudaTensor_data(state, bbox),
                               THCudaTensor_data(state, trans),
                               THCudaTensor_data(state, out),
                               THCudaTensor_data(state, top_count),
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

void dcn_v2_psroi_pooling_cuda_backward(THCudaTensor * out_grad, 
                                        THCudaTensor * input, THCudaTensor * bbox,
                                        THCudaTensor * trans, THCudaTensor * top_count,
                                        THCudaTensor * input_grad, THCudaTensor * trans_grad,
                                        const int no_trans,
                                        const float spatial_scale,
                                        const int output_dim,
                                        const int group_size,
                                        const int pooled_size,
                                        const int part_size,
                                        const int sample_per_part,
                                        const float trans_std)
{
    THArgCheck(THCudaTensor_isContiguous(state, out_grad), 0, "out_grad tensor has to be contiguous");
    THArgCheck(THCudaTensor_isContiguous(state, input), 1, "input tensor has to be contiguous");
    THCAssertSameGPU(THCudaTensor_checkGPU(state, 7, input, bbox, trans, out_grad, top_count,
                    input_grad, trans_grad));

    const int batch = THCudaTensor_size(state, input, 0);
    const int channels = THCudaTensor_size(state, input, 1);
    const int height = THCudaTensor_size(state, input, 2);
    const int width = THCudaTensor_size(state, input, 3);
    const int channels_trans = no_trans? 2 : THCudaTensor_size(state, trans, 1);

    const int num_bbox = THCudaTensor_size(state, bbox, 0);
    if (num_bbox != THCudaTensor_size(state, out_grad, 0))
        THError("Output shape and bbox number wont match: (%d vs %d).", 
                THCudaTensor_size(state, out_grad, 0), num_bbox);

    DeformablePSROIPoolBackwardAcc(THCState_getCurrentStream(state),
                                   THCudaTensor_data(state, out_grad),
                                   THCudaTensor_data(state, input),
                                   THCudaTensor_data(state, bbox),
                                   THCudaTensor_data(state, trans),
                                   THCudaTensor_data(state, top_count),
                                   THCudaTensor_data(state, input_grad),
                                   THCudaTensor_data(state, trans_grad),
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