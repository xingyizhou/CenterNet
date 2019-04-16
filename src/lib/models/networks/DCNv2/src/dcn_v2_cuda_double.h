// #ifndef DCN_V2_CUDA
// #define DCN_V2_CUDA

// #ifdef __cplusplus
// extern "C"
// {
// #endif

void dcn_v2_cuda_forward(THCudaDoubleTensor *input, THCudaDoubleTensor *weight,
                         THCudaDoubleTensor *bias, THCudaDoubleTensor *ones,
                         THCudaDoubleTensor *offset, THCudaDoubleTensor *mask,
                         THCudaDoubleTensor *output, THCudaDoubleTensor *columns,
                         int kernel_h, int kernel_w,
                         const int stride_h, const int stride_w,
                         const int pad_h, const int pad_w,
                         const int dilation_h, const int dilation_w,
                         const int deformable_group);
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
                          int deformable_group);

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
                                       const double trans_std);

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
                                        const double trans_std);


// #ifdef __cplusplus
// }
// #endif

// #endif