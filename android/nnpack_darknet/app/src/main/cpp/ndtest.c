
#include "darknet.h"

double test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh,char *outfile, int fullscreen)
{
    //list *options = read_data_cfg(datacfg);
   // char *name_list = option_find_str(options, "names", "names.list");
    //char **names = get_labels(name_list);

    //image **alphabet = load_alphabet();
    double  t0, t1;
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
    float nms=.2;
#ifdef NNPACK
    nnp_initialize();
	net.threadpool = pthreadpool_create(4);
#endif

    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return -1;
            strtok(input, "\n");
		}
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

        for(int i=0;i<5;i++)
		    network_predict(net, X);

        gettimeofday(&stop, 0);
		//printf("%s: Predicted in %ld ms.\n", input, (stop.tv_sec * 1000 + stop.tv_usec / 1000) - (start.tv_sec * 1000 + start.tv_usec / 1000));
        get_region_boxes(l, im.w, im.h, net.w, net.h, thresh, probs, boxes, 0, 0, hier_thresh, 1);
        if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        //else if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        //draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);
        if(outfile){

        }
        else{

#ifdef OPENCV
            cvNamedWindow("predictions", CV_WINDOW_NORMAL); 
            if(fullscreen){
                cvSetWindowProperty("predictions", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
            }
            show_image(im, "predictions");
            cvWaitKey(0);
            cvDestroyAllWindows();
#endif
        }

        free_image(im);
        free_image(sized);
        free(boxes);
        free_ptrs((void **)probs, l.w*l.h*l.n);
        if (filename) break;
    }
#ifdef NNPACK
	pthreadpool_destroy(net.threadpool);
	nnp_deinitialize();
#endif
    return ((stop.tv_sec * 1000 + stop.tv_usec / 1000) - (start.tv_sec * 1000 + start.tv_usec / 1000))/5.0;
}

double run_detector_android(char *cfg,char *weights)
{

    float thresh = 0.24;
    float hier_thresh = .5;


    char *datacfg = "coco.data";
    //char *cfg = "tykk3.cfg";
    //char *weights = "tykk3.weights";

    char *filename = "dog.jpg";

    return test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, 0, 0);

}
