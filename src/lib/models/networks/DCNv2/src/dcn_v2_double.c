#include <TH/TH.h>
#include <stdio.h>
#include <math.h>

void dcn_v2_forward(THDoubleTensor *input, THDoubleTensor *weight,
                    THDoubleTensor *bias, THDoubleTensor *ones,
                    THDoubleTensor *offset, THDoubleTensor *mask,
                    THDoubleTensor *output, THDoubleTensor *columns,
                    const int pad_h, const int pad_w,
                    const int stride_h, const int stride_w,
                    const int dilation_h, const int dilation_w,
                    const int deformable_group)
{
    printf("only implemented in GPU");
}
void dcn_v2_backward(THDoubleTensor *input, THDoubleTensor *weight,
                     THDoubleTensor *bias, THDoubleTensor *ones,
                     THDoubleTensor *offset, THDoubleTensor *mask,
                     THDoubleTensor *output, THDoubleTensor *columns,
                     THDoubleTensor *grad_input, THDoubleTensor *grad_weight,
                     THDoubleTensor *grad_bias, THDoubleTensor *grad_offset,
                     THDoubleTensor *grad_mask, THDoubleTensor *grad_output,
                     int kernel_h, int kernel_w,
                     int stride_h, int stride_w,
                     int pad_h, int pad_w,
                     int dilation_h, int dilation_w,
                     int deformable_group)
{
    printf("only implemented in GPU");
}