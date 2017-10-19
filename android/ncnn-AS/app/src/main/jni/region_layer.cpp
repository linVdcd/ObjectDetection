#include "region_layer.h"
#include <stdlib.h>
#include <math.h>
static inline float logistic_activate(float x){return 1./(1. + exp(-x));}


void activate_array(float *x, const int n)
{
    int i;
    for(i = 0; i < n; ++i){
        x[i] = logistic_activate(x[i]);
    }
}


int entry_index(int location, int entry,int w,int h,int coords,int classes)
{
    int n =   location / (w*h);
    int loc = location % (w*h);
    return n*w*h*(coords+classes+1) + entry*w*h + loc;
}

void softmax(float *input, int n, float temp, int stride, float *output)
{
    int i;
    float sum = 0;
    float largest = -__FLT_MAX__;
    for(i = 0; i < n; ++i){
        if(input[i*stride] > largest) largest = input[i*stride];
    }
    for(i = 0; i < n; ++i){
        float e = exp(input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}


void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    int g, b;
    for(b = 0; b < batch; ++b){
        for(g = 0; g < groups; ++g){
            softmax(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
        }
    }
}

//typedef struct{
 //   float x, y, w, h;
//}box;

void forward_act(float *input,float *output,int w,int h,int c,int classes,int coords,int num){
    //memcpy(output,input,w*h*c);
    for (int n=0;n<num;n++){
        int index = entry_index(n*w*h,0,w,h,coords,classes);
        activate_array(output+index,2*w*h);
        index = entry_index(n*w*h,coords,w,h,coords,classes);
        activate_array(output+index,w*h);
    }
    int index = entry_index(0,coords+1,w,h,coords,classes);
    softmax_cpu(input + index, classes, num, w*h*c/num, w*h, 1, w*h, 1, output + index);
}

box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / w;
    b.y = (j + x[index + 1*stride]) / h;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}
void correct_region_boxes(box *boxes, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = boxes[i];
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw);
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth);
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        boxes[i] = b;
    }
}
void get_region_boxes(int iw, int ih, int netw, int neth, float thresh, float **probs, box *boxes,int relative,int w,int h,int c,int classes,int coords,int num,float *biases,float *output)
{

    int i,j,n;
    float *predictions = output;

    for (i = 0; i < w*h; ++i){
        int row = i / w;
        int col = i % w;
        for(n = 0; n < num; ++n){
            int index = n*w*h + i;
            for(j = 0; j < classes; ++j){
                probs[index][j] = 0;
            }
            int obj_index = entry_index(n*w*h + i, coords,w,h,coords,classes);
            int box_index = entry_index(n*w*h + i, 0,w,h,coords,classes);
            float scale = predictions[obj_index];
            boxes[index] = get_region_box(predictions, biases, n, box_index, col, row, w, h, w*h);

            //int class_index = entry_index(n*w*h + i, coords + 1,w,h,coords,classes);

            float max = 0;
            for(j = 0; j < classes; ++j){
                int class_index = entry_index(n*w*h + i, coords + 1 + j,w,h,coords,classes);
                float prob = scale*predictions[class_index];
                probs[index][j] = (prob > thresh) ? prob : 0;
                if(prob > max) max = prob;
                // TODO REMOVE
                // if (j == 56 ) probs[index][j] = 0;
                /*
                   if (j != 0) probs[index][j] = 0;
                   int blacklist[] = {121, 497, 482, 504, 122, 518,481, 418, 542, 491, 914, 478, 120, 510,500};
                   int bb;
                   for (bb = 0; bb < sizeof(blacklist)/sizeof(int); ++bb){
                   if(index == blacklist[bb]) probs[index][j] = 0;
                   }
                 */
            }
            probs[index][classes] = max;


        }
    }
    correct_region_boxes(boxes, w*h*num, iw, ih, netw, neth, 1);
}


void region_forward(float * input,int iw,int ih,int nw,int nh,int w,int h,int c,int classes,int coords,int num,float nms,box *boxes, float **probs,float *biases){
    float *output;
    //float biases[10]={0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828};

    output = (float*)calloc(w*h*c,sizeof(float));

    memcpy(output,input,w*h*c);
    forward_act(input,output,w,h,c,classes,coords,num);
    //box *boxes = (box*)malloc(sizeof(box)*w*h*num);//calloc(w*h*num, sizeof(box));
    //float **probs = (float**)malloc(sizeof(float*)*w*h*num);//calloc(w*h*num, sizeof(float *));
    //for(int j = 0; j < w*h*num; ++j) probs[j] = (float*)malloc(sizeof(float)*(classes+1));//calloc(classes + 1, sizeof(float *));
    get_region_boxes(iw,ih,nw,nh,0.24,probs,boxes,0,w,h,c,classes,coords,num,biases,output);
    if (nms)
        do_nms_obj(boxes, probs, w*h*num,classes, nms);
    //free(boxes);
    //free(probs);
    free(output);
}