//
// Created by lin on 17-8-4.
//
#include "depthwise_conv2d_neon.h"
#include "dw.h"
#include "types.h"
#include <vector>
#include <arm_neon.h>
void SetNeonDimStrides(Dims<4>* d) {
    long int stride = 1;
    for (int i = 0; i < 4; ++i) {
        d->strides[i] = stride;
        stride *= d->sizes[i];
    }
}
void toDims(Dims<4>* d,int a[]){
    for (int i =0;i<4;i++)
        d->sizes[i]=a[i];
}





void dw_interface_test(){
    int batch=1;
    float input_data[48];//batch,h,w,in_channel
    int a[4] ={3,4,4,batch};//in_channel,h,w,batch
    Dims<4> input_dims; toDims(&input_dims,a);SetNeonDimStrides(&input_dims);
    float filter_data[27];//size,size,in_channel,depthmulti
    int b[4] ={3,3,3,1};//in_channel*depthmulti,size,size,1
    Dims<4> filter_dims; toDims(&filter_dims,b);SetNeonDimStrides(&filter_dims);
    float  bias_data[3]={0,0,0};//in_channel*depthmulti
    int c[4] = {3,1,1,1};//in_channel*depthmulti
    Dims<4> bias_dims; toDims(&bias_dims,c);SetNeonDimStrides(&bias_dims);
    int stride=1;
    int pad_width=1;
    int pad_height=1;
    int depth_multiplier=1;
    float output_data[48];//batch,oh,ow,in_channel*depthmulti
    int d[4]={3,4,4,batch};//in_channel*depthmulti,oh,ow
    Dims<4> output_dims; toDims(&output_dims,d); SetNeonDimStrides(&output_dims);
    float aa[1][4][4][3];

    int batchSize = 4*4*3;

    for(int b=0;b<batch;b++)
        for (int h = 0;h<4;h++)
            for(int w =0;w<4;w++)
                for (int c =0;c<3;c++)
                    input_data[b*batchSize+h*4*3+w*3+c]= float(c)+(float)1.0;

    for (int h=0;h<3;h++)
        for(int w=0;w<3;w++)
            for(int c=0;c<3;c++)
                filter_data[h*3*3+w*3+c] = float(c)+float(1.0);





    DepthwiseConv<FusedActivationFunctionType::kNone>(input_data, input_dims,
                                                      filter_data, filter_dims,
                                                      bias_data, bias_dims, stride,
                                                      pad_width, pad_height, depth_multiplier,
                                                      output_data, output_dims);


    for(int b=0;b<batch;b++)
        for (int h = 0;h<4;h++)
            for(int w =0;w<4;w++)
                for (int c =0;c<3;c++)
                    aa[b][h][w][c]=output_data[b*batchSize+h*4*3+w*3+c];
}


//void reshape()

void depthwise_conv2d_inference(float* input_data,float* filter_data,float* output_data,int input_shape[],int filter_shape[],int output_shape[],
                                int stride,int padding,int depth_multiplier){
/*
    input_data: [batch,h,w,c],
    filter_data: [size,size,in_channel,depth_multiplier],
    output_data: [batch,h,w,c],


    input_shape: [in_channel,h,w,batch],
    filter_shape: [in_channel*depthmulti,size,size,1],
    output_shape: [in_channel*depthmulti,oh,ow,batch],

    由于darknet的输入输出数据是[batch,channel,h,w],所以需要将输入转置成[batch,h,w,channel],将输出转置成[batch,channel,h,w]


 */


    Dims<4> input_dims; toDims(&input_dims,input_shape);SetNeonDimStrides(&input_dims);
    Dims<4> filter_dims; toDims(&filter_dims,filter_shape);SetNeonDimStrides(&filter_dims);
    float  *bias_data;//in_channel*depthmulti
    bias_data = (float*)calloc(input_shape[0]*depth_multiplier,sizeof(float));
    int bias_shape[4] = {input_shape[0]*depth_multiplier,1,1,1};
    Dims<4> bias_dims; toDims(&bias_dims,bias_shape);SetNeonDimStrides(&bias_dims);
    int pad_width=padding;
    int pad_height=padding;
    Dims<4> output_dims; toDims(&output_dims,output_shape); SetNeonDimStrides(&output_dims);

    DepthwiseConv<FusedActivationFunctionType::kNone>(input_data, input_dims,
                                                      filter_data, filter_dims,
                                                      bias_data, bias_dims, stride,
                                                      pad_width, pad_height, depth_multiplier,
                                                      output_data, output_dims);

    free(bias_data);
}