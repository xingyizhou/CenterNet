/*!
 * Copyright (c) 2017 Microsoft
 * Licensed under The MIT License [see LICENSE for details]
 * \file deformable_psroi_pooling.cu
 * \brief
 * \author Yi Li, Guodong Zhang, Jifeng Dai
*/
/***************** Adapted by Charles Shang *********************/

#ifndef DCN_V2_PSROI_POOLING_CUDA_DOUBLE
#define DCN_V2_PSROI_POOLING_CUDA_DOUBLE

#ifdef __cplusplus
extern "C"
{
#endif

    void DeformablePSROIPoolForward(cudaStream_t stream,
                                    const double *data,
                                    const double *bbox,
                                    const double *trans,
                                    double *out,
                                    double *top_count,
                                    const int batch,
                                    const int channels,
                                    const int height,
                                    const int width,
                                    const int num_bbox,
                                    const int channels_trans,
                                    const int no_trans,
                                    const double spatial_scale,
                                    const int output_dim,
                                    const int group_size,
                                    const int pooled_size,
                                    const int part_size,
                                    const int sample_per_part,
                                    const double trans_std);

    void DeformablePSROIPoolBackwardAcc(cudaStream_t stream,
                                        const double *out_grad,
                                        const double *data,
                                        const double *bbox,
                                        const double *trans,
                                        const double *top_count,
                                        double *in_grad,
                                        double *trans_grad,
                                        const int batch,
                                        const int channels,
                                        const int height,
                                        const int width,
                                        const int num_bbox,
                                        const int channels_trans,
                                        const int no_trans,
                                        const double spatial_scale,
                                        const int output_dim,
                                        const int group_size,
                                        const int pooled_size,
                                        const int part_size,
                                        const int sample_per_part,
                                        const double trans_std);

#ifdef __cplusplus
}
#endif

#endif