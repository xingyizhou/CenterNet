/*!
 * Copyright (c) 2017 Microsoft
 * Licensed under The MIT License [see LICENSE for details]
 * \file deformable_psroi_pooling.cu
 * \brief
 * \author Yi Li, Guodong Zhang, Jifeng Dai
*/
/***************** Adapted by Charles Shang *********************/

#ifndef DCN_V2_PSROI_POOLING_CUDA
#define DCN_V2_PSROI_POOLING_CUDA

#ifdef __cplusplus
extern "C"
{
#endif

    void DeformablePSROIPoolForward(cudaStream_t stream,
                                    const float *data,
                                    const float *bbox,
                                    const float *trans,
                                    float *out,
                                    float *top_count,
                                    const int batch,
                                    const int channels,
                                    const int height,
                                    const int width,
                                    const int num_bbox,
                                    const int channels_trans,
                                    const int no_trans,
                                    const float spatial_scale,
                                    const int output_dim,
                                    const int group_size,
                                    const int pooled_size,
                                    const int part_size,
                                    const int sample_per_part,
                                    const float trans_std);

    void DeformablePSROIPoolBackwardAcc(cudaStream_t stream,
                                        const float *out_grad,
                                        const float *data,
                                        const float *bbox,
                                        const float *trans,
                                        const float *top_count,
                                        float *in_grad,
                                        float *trans_grad,
                                        const int batch,
                                        const int channels,
                                        const int height,
                                        const int width,
                                        const int num_bbox,
                                        const int channels_trans,
                                        const int no_trans,
                                        const float spatial_scale,
                                        const int output_dim,
                                        const int group_size,
                                        const int pooled_size,
                                        const int part_size,
                                        const int sample_per_part,
                                        const float trans_std);

#ifdef __cplusplus
}
#endif

#endif