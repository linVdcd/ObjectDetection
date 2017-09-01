#include <jni.h>
#include <time.h>
#include <stdlib.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include "android_fs.h"
#include "darknet.h"

#include "depthwise_convolutional_layer.h"
#include "gemm.h"
double test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh,char *outfile, int fullscreen,int num)
{

    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);
    struct timeval start, stop;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.1;
    int ep=num;
#ifdef NNPACK
    nnp_initialize();
    net.threadpool = pthreadpool_create(4);
#endif


    strncpy(input, filename, 256);

#ifdef NNPACK
        image im = load_image_thread(input, 0, 0, net.c, net.threadpool);
        image sized = letterbox_image_thread(im, net.w, net.h, net.threadpool);
#else
        image im = load_image_color(input,0,0);
		image sized = letterbox_image(im, net.w, net.h);
		//image sized = resize_image(im, net.w, net.h);
		//image sized2 = resize_max(im, net.w);
		//image sized = crop_image(sized2, -((net.w - sized2.w)/2), -((net.h - sized2.h)/2), net.w, net.h);
		//resize_network(&net, sized.w, sized.h);
#endif
    layer l = net.layers[net.n-1];

    box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes + 1, sizeof(float *));

    float *X = sized.data;

    gettimeofday(&start, 0);

    for(int i=0;i<ep;i++)
        network_predict(net, X);

    gettimeofday(&stop, 0);
    double usetime =((stop.tv_sec * 1000 + stop.tv_usec / 1000) - (start.tv_sec * 1000 + start.tv_usec / 1000))/((float)(ep));
    //usetime /=double(ep);
    get_region_boxes(l, im.w, im.h, net.w, net.h, thresh, probs, boxes, 0, 0, hier_thresh, 1);
    if (nms)
        do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        //else if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        //draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);

    free_image(im);
    free_image(sized);
    free(boxes);
    free_ptrs((void **)probs, l.w*l.h*l.n);


#ifdef NNPACK
    pthreadpool_destroy(net.threadpool);
    nnp_deinitialize();
#endif
    return usetime;
}




double run_detector_android(char *cfg,char *weights,int num)
{

    float thresh = 0.24;
    float hier_thresh = .5;
    char *datacfg = "coco.data";
    char *filename = "dog.jpg";

    return test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, 0, 0,num);

}




jstring
Java_com_example_lin_nnpack_1darknet_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject thiz,
        jobject assetManager) {

//    test_depthwise_convolutional_layer();

    AAssetManager *mgr1 = AAssetManager_fromJava(env, assetManager);
    mgr=mgr1;
    char buffer[512];
    int num=1;//执行次数
    char*  str;
    double time,time1;

    // time =run_detector_android("ty9M.cfg","ty9M.backup");
    time =run_detector_android("Smallyolo3.cfg","Smallyolo3.weights",num);
     //time =run_detector_android("tykk3.cfg","tykk3.weights",num);
    // time =run_detector_android("ty-kk5.cfg","ty-kk5.backup");
    //time = run_detector_android("mobilenet_yolo.cfg","mobilenet_yolo.weights",num);
    asprintf(&str, "time: %g ms\n",time);

    /*float a[9];
    float b[389376];
    float c[43264];
    int n=43264;
    struct timeval t1,t2,t3;

    for(int i=0;i<9;i++)
    {
        a[i]=(float)(i);

    }
    for (int j=0;j<n*9;j++)
        b[j] = 0.1;
    gettimeofday(&t1, 0);
    for (int i=0;i<1000;i++)
        gemm(0, 0, 1, n, 9, 1, a, 9, b, n, 1, c,n);
    gettimeofday(&t2, 0);
    for (int i=0;i<1000;i++)
        VmulM(9,n,b,a,c);
    gettimeofday(&t3, 0);

    time = ((t2.tv_sec * 1000 + t2.tv_usec / 1000) - (t1.tv_sec * 1000 + t1.tv_usec / 1000))/((float)(1000));
    time1 = ((t3.tv_sec * 1000 + t3.tv_usec / 1000) - (t2.tv_sec * 1000 + t2.tv_usec / 1000))/((float)(1000));

    asprintf(&str, "time: %g ms\ntime: %g ms",time,time1);*/

    strlcpy(buffer, str, sizeof buffer);
    free(str);
    return (*env)->NewStringUTF(env, buffer);
}
