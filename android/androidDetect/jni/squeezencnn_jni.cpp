// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/bitmap.h>
#include <android/log.h>
#include "region_layer.h"
#include "image.h"
#include <string>
#include <vector>
#include <Android/Sdk/ndk-bundle/platforms/android-21/arch-arm/usr/include/android/bitmap.h>

// ncnn
#include "net.h"
#include "box.h"


static struct timeval tv_begin;
static struct timeval tv_end;
static double elasped;

static void bench_start()
{
    gettimeofday(&tv_begin, NULL);
}

static void bench_end(const char* comment)
{
    gettimeofday(&tv_end, NULL);
    elasped = ((tv_end.tv_sec - tv_begin.tv_sec) * 1000000.0f + tv_end.tv_usec - tv_begin.tv_usec) / 1000.0f;
//     fprintf(stderr, "%.2fms   %s\n", elasped, comment);
    __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "%.2fms   %s", elasped, comment);
}

static std::vector<unsigned char> squeezenet_param;
static std::vector<unsigned char> squeezenet_bin;
static std::vector<std::string> squeezenet_words;
static ncnn::Net squeezenet;

static std::vector<std::string> split_string(const std::string& str, const std::string& delimiter)
{
    std::vector<std::string> strings;

    std::string::size_type pos = 0;
    std::string::size_type prev = 0;
    while ((pos = str.find(delimiter, prev)) != std::string::npos)
    {
        strings.push_back(str.substr(prev, pos - prev));
        prev = pos + 1;
    }

    // To get the last substring (or only, if delimiter is not found)
    strings.push_back(str.substr(prev));

    return strings;
}

extern "C" {

// public native boolean Init(byte[] param, byte[] bin, byte[] words);
JNIEXPORT jboolean JNICALL Java_org_tensorflow_demo_SqueezeNcnn_Init(JNIEnv* env, jobject thiz, jbyteArray param, jbyteArray bin, jbyteArray words)
{
    // init param
    {
        int len = env->GetArrayLength(param);
        squeezenet_param.resize(len);
        env->GetByteArrayRegion(param, 0, len, (jbyte*)squeezenet_param.data());
        int ret = squeezenet.load_param(squeezenet_param.data());
        __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "load_param %d %d", ret, len);
    }

    // init bin
    {
        int len = env->GetArrayLength(bin);
        squeezenet_bin.resize(len);
        env->GetByteArrayRegion(bin, 0, len, (jbyte*)squeezenet_bin.data());
        int ret = squeezenet.load_model(squeezenet_bin.data());
        __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "load_model %d %d", ret, len);
    }


    // init words
    {
        int len = env->GetArrayLength(words);
        std::string words_buffer;
        words_buffer.resize(len);
        env->GetByteArrayRegion(words, 0, len, (jbyte*)words_buffer.data());
        squeezenet_words = split_string(words_buffer, "\n");
    }

    return JNI_TRUE;
}



void free_ptrs(void **ptrs, int n)
{
    int i;
    for(i = 0; i < n; ++i) free(ptrs[i]);
    free(ptrs);
}


// public native String Detect(Bitmap bitmap);
JNIEXPORT void JNICALL Java_org_tensorflow_demo_SqueezeNcnn_Detect(JNIEnv* env, jobject thiz,jobject bitmap,jfloatArray output)
{


    // ncnn from bitmap
    //AAssetManager *mgr1 = AAssetManager_fromJava(env, assetManager);
    //mgr=mgr1;

    //float data[416*416*3]={0};
    //FILE *f = android_fopen("size.txt","rb");
    //fread(data, sizeof(float),416*416*3,f);
    //fclose(f);



    /*int lenn = (env)->GetArrayLength(inputa);
     jfloat *datac  =env->GetFloatArrayElements(inputa,false);


    for(int i=0;i<lenn;i++)
        datacc[i]=datac[i];
    env->ReleaseFloatArrayElements(inputa,datac,0);
   ncnn::Mat in(416,416,3,datacc);*/



        ncnn::Mat in;

       AndroidBitmapInfo info;
        AndroidBitmap_getInfo(env, bitmap, &info);
        int width = info.width;
        int height = info.height;

        int inputsize=320;
        void* indata;
        AndroidBitmap_lockPixels(env, bitmap, &indata);

        in = ncnn::Mat::from_pixels((const unsigned char*)indata, ncnn::Mat::PIXEL_RGBA2RGB, width, height);

        AndroidBitmap_unlockPixels(env, bitmap);
    /*for(int w=0;w<416;w++)
        for(int h=0;h<416;h++)
            for(int c=0;c<3;c++)
            {
                datacc[c*416*416+w*416+h] = in.data[w*416*3+h*3+c]/256.f;

            }*/
    image im;
    for(int i=0;i<width*height*3;i++)
        in.data[i] /=256.f;

    im.data = in.data;
    im.w=width;
    im.h=height;
    im.c=3;
    image sized = letterbox_image(im, inputsize, inputsize);


    ncnn::Mat inresize(inputsize,inputsize,3,sized.data);
    //
    ncnn::Mat out;
    bench_start();

    ncnn::Extractor ex=squeezenet.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
    ex.input(0, inresize);
    ex.extract(209, out);
    bench_end("detect");
    int outsize=10;
    ncnn::Mat out1(out.reshape(outsize*outsize*425));



    int iw=width,ih=height,nw=inputsize,nh=inputsize,w=outsize,h=outsize,c=425,num=5,classes=80,strid=85;
    box *boxes = (box*)malloc(sizeof(box)*w*h*num);//calloc(w*h*num, sizeof(box));
    float **probs = (float**)malloc(sizeof(float*)*w*h*num);//calloc(w*h*num, sizeof(float *));
    for(int j = 0; j < w*h*num; ++j) probs[j] = (float*)malloc(sizeof(float)*(classes+1));//calloc(classes + 1, sizeof(float *));
    float biases[10]={0.738768,0.874946,  2.42204,2.65704,  4.30971,7.04493,  10.246,4.59428,  12.6868,11.8741};
    region_forward(out1.data,iw,ih,nw,nh,w,h,c,classes,4,num,0.1,boxes,probs,biases);

    jfloat *outputc;
    outputc = env->GetFloatArrayElements(output, false);
    for(int i = 0;i<w*h*num;i++)
    {
        outputc[i*strid+0] =boxes[i].x;
        outputc[i*strid+1] =boxes[i].y;
        outputc[i*strid+2] =boxes[i].w;
        outputc[i*strid+3] =boxes[i].h;
        for(int j=0;j<81;j++)
            outputc[i*strid+j+4] = probs[i][j];
    }


    free_image(sized);
    free(boxes);
    free_ptrs((void **)probs, w*h*num);

    env->SetFloatArrayRegion(output,0,outsize*outsize*425,outputc);

    env->ReleaseFloatArrayElements(output,outputc,0);

}

}
