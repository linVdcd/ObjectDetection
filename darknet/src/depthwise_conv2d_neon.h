//
// Created by lin on 17-8-4.
//

#ifndef HELLO_NEON_INTERFACE1_H
#define HELLO_NEON_INTERFACE1_H
#ifdef __cplusplus
extern "C" {
#endif
 void dw_interface_test();
void depthwise_conv2d_inference(float* input_data,float* filter_data,float* output_data,int input_shape[],int filter_shape[],int output_shape[],
                                int stride,int padding,int depth_multiplier);

#ifdef __cplusplus
}
#endif
#endif //HELLO_NEON_INTERFACE_H
