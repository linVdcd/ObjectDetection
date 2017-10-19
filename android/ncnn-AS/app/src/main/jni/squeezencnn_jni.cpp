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
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <jni.h>

#include <string>
#include <vector>
#include "android_fs.h"
// ncnn
#include "net.h"
#include "region_layer.h"


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
JNIEXPORT jboolean JNICALL Java_com_tencent_squeezencnn_SqueezeNcnn_Init(JNIEnv* env, jobject thiz, jbyteArray param, jbyteArray bin, jbyteArray words)
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
JNIEXPORT jstring JNICALL Java_com_tencent_squeezencnn_SqueezeNcnn_Detect(JNIEnv* env, jobject thiz, jobject assetManager)
{


    // ncnn from bitmap
    AAssetManager *mgr1 = AAssetManager_fromJava(env, assetManager);
    mgr=mgr1;

    float data[416*416*3]={0};
    FILE *f = android_fopen("size.txt","rb");
    fread(data, sizeof(float),416*416*3,f);
    fclose(f);




    ncnn::Mat in(416,416,3,data);




    /*{
        AndroidBitmapInfo info;
        AndroidBitmap_getInfo(env, bitmap, &info);
        int width = info.width;
        int height = info.height;
        if (width != 416 || height != 416)
            return NULL;
        if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
            return NULL;

        void* indata;
        AndroidBitmap_lockPixels(env, bitmap, &indata);

        in = ncnn::Mat::from_pixels((const unsigned char*)indata, ncnn::Mat::PIXEL_RGBA2RGB, width, height);

        AndroidBitmap_unlockPixels(env, bitmap);
    }*/

    ncnn::Mat out;
    bench_start();
    for(int i=0;i<20;i++){
    ncnn::Extractor ex = squeezenet.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
    ex.input(0, in);
    ex.extract(48, out);}
    bench_end("detect");
    int iw=768,ih=576,nw=416,nh=416,w=13,h=13,c=425,num=5,classes=80;
    ncnn::Mat out1(out.reshape(13*13*425));
    box *boxes = (box*)malloc(sizeof(box)*w*h*num);//calloc(w*h*num, sizeof(box));
    float **probs = (float**)malloc(sizeof(float*)*w*h*num);//calloc(w*h*num, sizeof(float *));
    for(int j = 0; j < w*h*num; ++j) probs[j] = (float*)malloc(sizeof(float)*(classes+1));//calloc(classes + 1, sizeof(float *));
    float biases[10]={0.738768,0.874946,  2.42204,2.65704,  4.30971,7.04493,  10.246,4.59428,  12.6868,11.8741};
    region_forward(out1.data,iw,ih,nw,nh,w,h,c,classes,4,num,0.1,boxes,probs,biases);

    free(boxes);
    free_ptrs((void **)probs, w*h*num);

    const std::string& word = "time";
    char tmp[32];
    sprintf(tmp, "%.3f", elasped/(float(20)));
    std::string result_str = std::string(word.c_str()) + " = " + tmp;

    // +10 to skip leading n03179701
    jstring result = env->NewStringUTF(result_str.c_str());



    return result;
}

}
