#ifndef REGION_LAYER_H
#define REGION_LAYER_H

#include "box.h"
void region_forward(float * input,int iw,int ih,int nw,int nh,int w,int h,int c,int classes,int coords,int num,float nms,box *boxes, float **probs,float *biases);

#endif