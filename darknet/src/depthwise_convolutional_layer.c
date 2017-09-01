#include "depthwise_convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"

#ifdef HAVE_TF_DW
#include "depthwise_conv2d_neon.h"
#endif





#ifdef HAVE_NEON

#include <arm_neon.h>

#endif



#ifdef HAVE_TF_DW

struct reshape_context{
    float *output;
    float *data;
    int channel_size;
    int Wchannel;
    int W;
    int channel;
};

static void nchw2nhwc(struct reshape_context* context, int index){
    int
        c = index/context->channel_size, //H*W
         tmp = index%context->channel_size,
            h = tmp/context->W,//W
            w = tmp%context->W;
        context->output[h*context->Wchannel+w*context->channel+c] = context->data[index];
}

static void nhwc2nchw(struct reshape_context* context, int index)
{
    int
            c = index/context->channel_size, //H*W
            tmp = index%context->channel_size,
            h = tmp/context->W,//W
            w = tmp%context->W;
    context->output[index] = context->data[h*context->Wchannel+w*context->channel+c];
}


void NCHW2NHWC_thread(float * data,int H,int W,int channel,pthreadpool_t threadpool){
    int size = channel*H*W;
    float *output = (float*)calloc(size,sizeof(float));
    struct reshape_context context = {
            .output=output,
            .data =data,
            .channel_size=H*W,
            .Wchannel=W*channel,
            .W = W,
            .channel=channel,
    };
    pthreadpool_compute_1d_tiled(threadpool,(pthreadpool_function_1d_tiled_t) nchw2nhwc,(void**) &context,channel*H*W,1);
    memcpy(context.data,output,size*sizeof(float));
    free(output);
}

void NHWC2NCHW_thread(float* data,int H,int W,int channel,pthreadpool_t threadpool){
    int size = channel*H*W;
    float *output = (float*)calloc(size,sizeof(float));
    struct reshape_context context = {
            .output=output,
            .data =data,
            .channel_size=H*W,
            .Wchannel=W*channel,
            .W = W,
            .channel=channel,
    };
    pthreadpool_compute_1d_tiled(threadpool,(pthreadpool_function_1d_tiled_t) nhwc2nchw,(void**) &context,channel*H*W,1);
    memcpy(context.data,output,size*sizeof(float));
    free(output);
}



void NCHW2NHWC(float *data ,float*output, int dims[])
{
    int size = dims[0]*dims[1]*dims[2]*dims[3];
    //float *output = (float*)calloc(size,sizeof(float));
    int batch=dims[0],channel=dims[1],H=dims[2],W=dims[3]
    ,bs = channel*H*W ,wc = W*channel,hw=H*W;
    for (int b=0;b<batch;b++)
        for(int c=0;c<channel;c++)
            for(int h=0;h<H;h++)
                for(int w=0;w<W;w++)
                    output[b*bs+h*wc+w*channel+c] = data[b*bs+c*hw+h*W+w];
    memcpy(data,output,size*sizeof(float));
    //free(output);
}

void NHWC2NCHW(float *data,float*output,int dims[]){
    int size = dims[0]*dims[1]*dims[2]*dims[3];
    //float *output = (float*)calloc(size,sizeof(float));
    int batch=dims[0],channel=dims[3],H=dims[1],W=dims[2],
    bs = channel*H*W ,wc = W*channel,hw=H*W;
    for(int b=0;b<batch;b++)
        for(int h=0;h<H;h++)
            for(int w=0;w<W;w++)
                for(int c=0;c<channel;c++)
                    output[b*bs+c*hw+h*W+w] = data[b*bs+h*wc+w*channel+c];
    memcpy(data,output,size*sizeof(float));
    //free(output);
}

#endif


int depthwise_convolutional_out_height(depthwise_convolutional_layer l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

int depthwise_convolutional_out_width(depthwise_convolutional_layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}


//��ʱ���ݿռ���?
static size_t get_workspace_size(layer l){
#ifdef CUDNN
    if(gpu_index >= 0){
        size_t most = 0;
        size_t s = 0;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.weightDesc,
                l.convDesc,
                l.dstTensorDesc,
                l.fw_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dweightDesc,
                l.bf_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
                l.weightDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dsrcTensorDesc,
                l.bd_algo,
                &s);
        if (s > most) most = s;
        return most;
    }
#endif
    return (size_t)l.h*l.w*l.size*l.size*l.c*sizeof(float);
}




#ifdef GPU
#ifdef CUDNN
void cudnn_depthwise_convolutional_setup(layer *l)
{
    cudnnSetTensor4dDescriptor(l->dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w);
    cudnnSetTensor4dDescriptor(l->ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
    cudnnSetFilter4dDescriptor(l->dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, l->c, l->size, l->size);

    cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
    cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 
    cudnnSetFilter4dDescriptor(l->weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, l->c, l->size, l->size);
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION);
    /*cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->weightDesc,
            l->convDesc,
            l->dstTensorDesc,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            0,
            &l->fw_algo);
    cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(),
            l->weightDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dsrcTensorDesc,
            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
            0,
            &l->bd_algo);
    cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dweightDesc,
            CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
            0,
            &l->bf_algo);*/
}
#endif
#endif

depthwise_convolutional_layer make_depthwise_convolutional_layer(int batch, int h, int w, int c,int size, int stride, int padding, ACTIVATION activation, int batch_normalize)
{
    int i;
	depthwise_convolutional_layer l = {0};
    l.type = DEPTHWISE_CONVOLUTIONAL;

    l.h = h;
    l.w = w;
    l.n = c;
	l.c = c;

    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;

    l.weights = calloc(l.n*size*size, sizeof(float));
    /*
    for(int cc = 0;cc<l.n;cc++)
        for (int s1=0;s1<size;s1++)
            for(int s2=0;s2<size;s2++)
                l.weights[cc*size*size+s1*size+s2] = (float)(cc+1);*/

    l.weight_updates = calloc(l.n*size*size, sizeof(float));

    l.biases = calloc(l.n, sizeof(float));
    l.bias_updates = calloc(l.n, sizeof(float));

    l.nweights = l.n*size*size;
    l.nbiases = l.n;

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c));
    //scale = .02;
   //for(i = 0; i < c*size*size; ++i) l.weights[i] = 0.01*i;
    //for(i = 0; i < l.n*l.size*l.size; ++i) l.weights[i] = scale*rand_normal();
    int out_w = depthwise_convolutional_out_width(l);
    int out_h = depthwise_convolutional_out_height(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = l.n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

    l.forward = forward_depthwise_convolutional_layer;
    l.backward = backward_depthwise_convolutional_layer;
    l.update = update_depthwise_convolutional_layer;


    if(batch_normalize){
        l.scales = calloc(c, sizeof(float));
        l.scale_updates = calloc(c, sizeof(float));
        for(i = 0; i < c; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(c, sizeof(float));
        l.variance = calloc(c, sizeof(float));

        l.mean_delta = calloc(c, sizeof(float));
        l.variance_delta = calloc(c, sizeof(float));

        l.rolling_mean = calloc(c, sizeof(float));
        l.rolling_variance = calloc(c, sizeof(float));
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }


#ifdef GPU
    l.forward_gpu = forward_depthwise_convolutional_layer_gpu;
    l.backward_gpu = backward_depthwise_convolutional_layer_gpu;
    l.update_gpu = update_depthwise_convolutional_layer_gpu;

    if(gpu_index >= 0){


        l.weights_gpu = cuda_make_array(l.weights, c*size*size);
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, c*size*size);

        l.biases_gpu = cuda_make_array(l.biases, c);
        l.bias_updates_gpu = cuda_make_array(l.bias_updates, c);

        l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*c);
        l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*c);



        if(batch_normalize){
            l.mean_gpu = cuda_make_array(l.mean, c);
            l.variance_gpu = cuda_make_array(l.variance,c);

            l.rolling_mean_gpu = cuda_make_array(l.mean, c);
            l.rolling_variance_gpu = cuda_make_array(l.variance, c);

            l.mean_delta_gpu = cuda_make_array(l.mean, c);
            l.variance_delta_gpu = cuda_make_array(l.variance, c);

            l.scales_gpu = cuda_make_array(l.scales, c);
            l.scale_updates_gpu = cuda_make_array(l.scale_updates, c);

            l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*c);
            l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*c);
        }
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.srcTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnCreateFilterDescriptor(&l.weightDesc);
        cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
        cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
        cudnnCreateFilterDescriptor(&l.dweightDesc);
        cudnnCreateConvolutionDescriptor(&l.convDesc);
        cudnn_depthwise_convolutional_setup(&l);
#endif
    }
#endif
    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    fprintf(stderr, "depthwise conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", c, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);



    return l;
}

void resize_depthwise_convolutional_layer(depthwise_convolutional_layer *l, int w, int h)
{
	l->w = w;
	l->h = h;
	int out_w = depthwise_convolutional_out_width(*l);
	int out_h = depthwise_convolutional_out_height(*l);

	l->out_w = out_w;
	l->out_h = out_h;

	l->outputs = l->out_h * l->out_w * l->out_c;
	l->inputs = l->w * l->h * l->c;

	l->output = realloc(l->output, l->batch*l->outputs * sizeof(float));
	l->delta = realloc(l->delta, l->batch*l->outputs * sizeof(float));
	if (l->batch_normalize) {
		l->x = realloc(l->x, l->batch*l->outputs * sizeof(float));
		l->x_norm = realloc(l->x_norm, l->batch*l->outputs * sizeof(float));
	}

#ifdef GPU
	cuda_free(l->delta_gpu);
	cuda_free(l->output_gpu);

	l->delta_gpu = cuda_make_array(l->delta, l->batch*l->outputs);
	l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);

	if (l->batch_normalize) {
		cuda_free(l->x_gpu);
		cuda_free(l->x_norm_gpu);

		l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
		l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
	}
#ifdef CUDNN
	cudnn_depthwise_convolutional_setup(l);
#endif
#endif
	l->workspace_size = get_workspace_size(*l);
}


void test_depthwise_convolutional_layer()
{
#include "softmax_layer.h"
#include "avgpool_layer.h"
#include "cost_layer.h"

    float data[] = {1,1,1,1,1,
    1,1,1,1,1,
    1,1,1,1,1,
                    1,1,1,1,1,
                    1,1,1,1,1,
    2,2,2,2,2,
                    2,2,2,2,2,
                    2,2,2,2,2,
    2,2,2,2,2,
    2,2,2,2,2,
    3,3,3,3,3,
                    3,3,3,3,3,
                    3,3,3,3,3,
    3,3,3,3,3,
    3,3,3,3,3};
	float truth[] = { 0,0,1 };
	float delta[27] = {0 };


	int num_layer = 4;
    int i;
	network net = make_network(num_layer);
	net.h=5;
	net.w=5;
	net.c=3;
	net.batch = 1;
	
	net.input = data;

	net.truth = truth;
	net.train = 1;



	depthwise_convolutional_layer depthwise_conv1 = make_depthwise_convolutional_layer(net.batch, net.h, net.w, net.c, 3, 1, 0, RELU, 0);


	avgpool_layer global_avgpool1 = make_avgpool_layer(net.batch, depthwise_conv1.out_w, depthwise_conv1.out_h, depthwise_conv1.n);
	softmax_layer softmax_1 = make_softmax_layer(net.batch, depthwise_conv1.n, 1);
	softmax_1.temperature = 1;//����ȱ��
	cost_layer cost_1 = make_cost_layer(net.batch, depthwise_conv1.n, SSE, 1);


	net.layers[0] = depthwise_conv1;
	net.layers[1] = global_avgpool1;
	net.layers[2] = softmax_1;
	net.layers[3] = cost_1;
	net.workspace = calloc(1, 75);



	for (i = 0; i < net.n; ++i) {
		net.index = i;
		layer l = net.layers[i];
		if (l.delta) {
			fill_cpu(l.outputs * l.batch, 0, l.delta, 1);
		}

		l.forward(l, net);
		net.input = l.output;
		if (l.truth) {
			net.truth = l.output;
		}
	}
	calc_network_cost(net);
	float cost = *net.cost;
	fprintf(stderr, "**********************cost:%f ***************", *net.cost);




	fprintf(stderr, "**********************backward *************** \n");


	network orig = net;
	for (i = net.n - 1; i >= 0; --i) {
		layer l = net.layers[i];
		if (i == 0) {
			//net = orig;
			net.input = data;
			net.delta = delta;
		}
		else {
			layer prev = net.layers[i - 1];
			net.input = prev.output;
			net.delta = prev.delta;//�м������ָ�븳ֵ����������backward��ʱ����ʵ�Ǹ����˵�ǰ�����ǰ��һ�����������?
		}
		net.index = i;
		l.backward(l, net);
	}


	

    //forward_depthwise_convolutional_layer(l,net);
}



void add_bias_depthwise(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias_depthwise(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void backward_bias_depthwise(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}

/*void ckk2kkc(float *data ,int K,int channel){// added by lin ming'an
    int size = K*K*channel;
    float *output = (float*)calloc(size,sizeof(float));
    for (int c=0;c<channel;c++)
        for (int k1=0;k1<K;k1++)
            for(int k2=0;k2<K;k2++)
                output[k1*K*channel+k2*channel+c] = data[c*K*K+k1*K+k2];
    memcpy(data,output,size*sizeof(float));
    free(output);
}*/







#ifdef HAVE_NEON // added by linmingan
struct dw_context{
    network net;
    depthwise_convolutional_layer l;

    int n;
    int imagesize;
    int k;

};
static void VmulM(int K, int input_depth,
                  float* input_ptr,
                  float* filter_ptr, float* acc_buffer_ptr) {

    float f=0,f2=0,f3=0,f4=0,*local_output_ptr=NULL,
     input_val=0,input_val1=0,input_val2=0,input_val3=0;
    int depth=0,num_block= K/4;
    const float* local_filter_ptr = filter_ptr,
            *local_input_ptr=NULL,*local_input_ptr1=NULL,
            *local_input_ptr2=NULL,*local_input_ptr3=NULL;

    float32x4_t acc,filter,filter1,filter2,filter3,
                input,input1,input2,input3;


    for (int k = 0; k < num_block; k++) {
        local_output_ptr = acc_buffer_ptr;
        local_input_ptr = input_ptr+k*4*input_depth;
        local_input_ptr1 = input_ptr+(k*4+1)*input_depth;
        local_input_ptr2 = input_ptr+(k*4+2)*input_depth;
        local_input_ptr3 = input_ptr+(k*4+3)*input_depth;

        f=*(local_filter_ptr);f2=*(local_filter_ptr+1);f3=*(local_filter_ptr+2);f4=*(local_filter_ptr+3);
        depth=0;
         filter = vdupq_n_f32(f);
         filter1 = vdupq_n_f32(f2);
         filter2 = vdupq_n_f32(f3);
         filter3 = vdupq_n_f32(f4);


        while (depth<input_depth-4) {
            acc = vld1q_f32(local_output_ptr);
            input = vld1q_f32(local_input_ptr);
            input1 = vld1q_f32(local_input_ptr1);
            input2 = vld1q_f32(local_input_ptr2);
            input3 = vld1q_f32(local_input_ptr3);

            acc= vmlaq_f32(acc,filter,input);
            acc= vmlaq_f32(acc,filter1,input1);
            acc= vmlaq_f32(acc,filter2,input2);
            acc= vmlaq_f32(acc,filter3,input3);

            vst1q_f32(local_output_ptr, acc);

            local_input_ptr += 4;
            local_input_ptr1 += 4;
            local_input_ptr2 += 4;
            local_input_ptr3 += 4;
            local_output_ptr+=4;
            depth+=4;
        }
        while(depth<input_depth){
            input_val = *local_input_ptr;
            input_val1 = *local_input_ptr1;
            input_val2 = *local_input_ptr2;
            input_val3 = *local_input_ptr3;
            *local_output_ptr += f * input_val;
            *local_output_ptr += f2 * input_val1;
            *local_output_ptr += f3 * input_val2;
            *local_output_ptr += f4 * input_val3;

            local_input_ptr++;
            local_input_ptr1++;
            local_input_ptr2++;
            local_input_ptr3++;

            local_output_ptr++;
            depth++;

        }
        local_filter_ptr+=4;
    }





    for (int k = num_block*4; k < K; k++) {
        local_output_ptr = acc_buffer_ptr;
        local_input_ptr = input_ptr+k*input_depth;

        f=*local_filter_ptr;
         filter = vdupq_n_f32(f);
        depth=0;
        while (depth<input_depth-4) {
            acc = vld1q_f32(local_output_ptr);
            input = vld1q_f32(local_input_ptr);
            acc= vmlaq_f32(acc,filter,input);
            vst1q_f32(local_output_ptr, acc);
            local_input_ptr += 4;
            local_output_ptr+=4;
            depth+=4;
        }
        while(depth<input_depth){
            input_val = *local_input_ptr;

            *local_output_ptr += f * input_val;
            local_input_ptr++;
            local_output_ptr++;
            depth++;

        }
        local_filter_ptr++;
    }
}

static void dw_thread( struct dw_context * context,int c){//added by linmingan

    float *aoffset = context->l.weights+c*context->k;
    float *boffset = context->net.workspace+c*context->k*context->imagesize;
    float *coffset = context->l.output+c*context->n;
    float *intput_offset = context->net.input + c*context->imagesize;
    im2col_cpu(intput_offset, 1, context->l.h, context->l.w,
               context->l.size, context->l.stride, context->l.pad, boffset);
    //gemm(0, 0, 1, context->n, context->k, 1, aoffset, context->k, boffset, context->n, 1, coffset,context->n);
    VmulM(context->k,context->n,boffset,aoffset,coffset);

}



void depthwise_conv2d_thread(network net,depthwise_convolutional_layer l,int n,int k,int imagesize,pthreadpool_t threadpool){
//added by linmingan

    struct dw_context contex={
            .net = net,
            .l=l,
            .n=n,
            .k=k,
            .imagesize=imagesize

    };

    pthreadpool_compute_1d(threadpool,(pthreadpool_function_1d_t) dw_thread,(void **)&contex,(size_t)l.c);

}
#endif


void forward_depthwise_convolutional_layer(depthwise_convolutional_layer l, network net)
{
    int out_h = l.out_h;
    int out_w = l.out_w;


    fill_cpu(l.outputs*l.batch, 0, l.output, 1);


    int k = l.size*l.size;
    int n = out_h*out_w;


#ifdef HAVE_TF_DW // added by  lin ming'an

    int darknet_input_shape[4]={l.batch,l.c,l.h,l.w},
        neon_input_shape[4]={l.c,l.h,l.w,l.batch},
        tt_output_shape[4]={l.batch,l.out_h,l.out_w,l.out_c},
        neon_output_shape[4] = {l.out_c,l.out_h,l.out_w,l.batch},
        neon_filter_shape[4]={l.n,l.size,l.size,1};

    //ckk2kkc(l.weights,l.size,l.n);

    NCHW2NHWC(net.input,net.workspace,darknet_input_shape);

    //NCHW2NHWC_thread(net.input,l.h,l.w,l.c,net.threadpool);//非常慢


    depthwise_conv2d_inference(net.input,
                               l.weights,
                               l.output,
                               neon_input_shape,
                               neon_filter_shape,
                               neon_output_shape,
                               l.stride,
                               l.pad,1);
    NHWC2NCHW(l.output,net.workspace,tt_output_shape);



    //NHWC2NCHW_thread(l.output,l.out_h,l.out_w,l.c,net.threadpool);

#elif HAVE_NEON // added by  lin ming'an
    int imagesize=l.w*l.h;
    depthwise_conv2d_thread(net,l,n,k,imagesize,net.threadpool);//im2col+多线程+neon

#else

    int i, b, c;
    for(b = 0; b < l.batch; ++b){
        for (c=0;c<l.c;c++)
        {
            float *aoffset = l.weights+c*l.size*l.size;
            float *boffset = net.workspace;
            float *coffset = l.output+c*l.out_h*l.out_w+b*l.n*l.out_h*l.out_w;
            float *intput_offset = net.input + c*l.h*l.w+ b*l.c*l.h*l.w;
            im2col_cpu(intput_offset, 1, l.h, l.w,
                       l.size, l.stride, l.pad, boffset);
            gemm(0, 0, 1, n, k, 1, aoffset, k, boffset, n, 1, coffset, n);


        }
    }
#endif


    if(l.batch_normalize){
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);
    }

	int m = l.n;
    activate_array(l.output, m*n*l.batch, l.activation);//�����ǰ�򴫵�
/*
	for (int i = 0; i < l.batch*l.c*l.out_h*l.out_w; i++)
	{
		fprintf(stderr, "%f \t", l.output[i]);
	}*/

}

void backward_depthwise_convolutional_layer(depthwise_convolutional_layer l, network net)
{
    int i, b, c;
    int m = l.n;
    int n = l.size*l.size;
    int k = l.out_w*l.out_h;
	//�����������
    gradient_array(l.output, m*k*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    }


	for (b = 0; b < l.batch; ++b) {
		for (c = 0; c<l.c; c++)
		{


			//��Ȩ����
			float *aoffset = l.delta + c*l.out_h*l.out_w + b*l.n*l.out_h*l.out_w;
			float *boffset = net.workspace;
			float *coffset = l.weight_updates + c*l.size*l.size;


			float *im = net.input + c*l.h*l.w + b*l.c*l.h*l.w;


			im2col_cpu(im, 1, l.h, l.w,
				l.size, l.stride, l.pad, boffset);
			gemm(0, 1, 1, n, k, 1, aoffset, k, boffset, k, 1, coffset, n);
			//�Ա������������󵼣�Ҳ������ԭʼ��Ȩ�أ������������΢������ͼ���о������

			if (net.delta) {
				aoffset = l.weights+ c*l.size*l.size;
				boffset = l.delta + c*l.out_h*l.out_w + b*l.n*l.out_h*l.out_w;
				coffset = net.workspace;

				gemm(1, 0, n, k, 1, 1, aoffset, n, boffset, k, 0, coffset, k);

				col2im_cpu(net.workspace, 1, l.h, l.w, l.size, l.stride, l.pad, net.delta + c*l.h*l.w + b*l.n*l.h*l.w);
			}


		}
	}


/*
	for (int i = 0; i < l.c*l.size*l.size; i++)
	{
		fprintf(stderr, "weight_updates:%f \t", l.weight_updates[i]);
	}
*/



}

void update_depthwise_convolutional_layer(depthwise_convolutional_layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    int size = l.size*l.size*l.c;
    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales){
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(size, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(size, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(size, momentum, l.weight_updates, 1);
}
void denormalize_depthwise_convolutional_layer(depthwise_convolutional_layer l)
{
	int i, j;
	for (i = 0; i < l.n; ++i) {
		float scale = l.scales[i] / sqrt(l.rolling_variance[i] + .00001);
		for (j = 0; j < l.size*l.size; ++j) {
			l.weights[i*l.size*l.size + j] *= scale;
		}
		l.biases[i] -= l.rolling_mean[i] * scale;
		l.scales[i] = 1;
		l.rolling_mean[i] = 0;
		l.rolling_variance[i] = 1;
	}
}








