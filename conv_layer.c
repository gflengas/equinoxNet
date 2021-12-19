#include "conv_layer.h"
#include "custom_math.h"
#include <stdio.h>

conv_layer init_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding){
    conv_layer l={0};
    //input
    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n; //number of filters
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;

    l.weights = calloc(c*n*size*size, sizeof(float));
    l.nweights = c*n*size*size;
    l.weight_updates = calloc(c*n*size*size, sizeof(float));

    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));
    l.nbiases = n;

    //initialise weights based on the He initialization
    float scale = sqrt(2./(l.nweights));
    for(int i = 0; i < l.nweights; ++i) l.weights[i] = scale* rand_uniform(-1,1);//rand_normal();
    //output
    l.out_h = (l.h + 2*l.pad - l.size) / l.stride + 1;
    l.out_w = (l.w + 2*l.pad - l.size) / l.stride + 1;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

    l.workspace_size = (size_t)l.out_h*l.out_w*l.size*l.size*l.c*sizeof(float);
    return l;
}

void conv_fwd(conv_layer l, network net){
    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = l.out_w*l.out_h;
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    for (int i = 0; i < l.batch; i++)
    {
        float *b = net.workspace;
        float *c = l.output + i*n*m;
        float *im =  net.input + i*l.h*l.w*l.c;
        //get the image adding padding and stride
        if (l.size == 1) {
            b = im;
        } else {
            im2col_cpu(im, l.c, l.h, l.w, l.size, l.stride, l.pad, b);
        }

        gemm_nn(m,n,k,l.weights,k,b,n,c,n);

        //gemm_nn(a,k,b,n,c,n);
    }
    //add biases
    int windowSize = l.out_h*l.out_w;
    for(int b = 0; b < l.batch; ++b){
        for(int i = 0; i < l.n; ++i){
            for(int j = 0; j < windowSize; ++j){
                l.output[(b*l.n + i)*windowSize + j] += l.biases[i];
            }
        }
    }
    //activation function using leaky relu
    for (int i = 0; i < l.outputs*l.batch; i++)
    {
        l.output[i] = leaky_activate(l.output[i]);
    }
}


void conv_bwd(conv_layer l, network net)
{
    int m = l.n;
    int n = l.size*l.size*l.c;
    int k = l.out_w*l.out_h;
    for (int i = 0; i < l.nweights; ++i) {
        l.weight_updates[i]=0;
    }
    for (int i = 0; i < l.nbiases; ++i) {
        l.bias_updates[i]=0;
    }
    //leaky relu gradient calculation
    for (int i = 0; i < l.outputs*l.batch; i++)
    {
        l.delta[i] *=leaky_gradient(l.output[i]);
    }

    //bw biases
    for (int b = 0; b < l.batch; b++)
    {
        for (int i = 0; i < l.n; i++)
        {
            l.bias_updates[i] += sum_array(l.delta+(i+b*l.n),k);
        }

    }

    for(int i = 0; i < l.batch; ++i){

        float *a = l.delta + i*m*k;
        float *b = net.workspace;
        float *c = l.weight_updates;

        float *im  = net.input + i*l.c*l.h*l.w;
        float *imd = net.delta + i*l.c*l.h*l.w;

        if(l.size == 1){
            b = im;
        } else {
            im2col_cpu(im, l.c, l.h, l.w,
                       l.size, l.stride, l.pad, b);
        }

        gemm_nt(m,n,k,a,k,b,k,c,n);

        if (net.delta) {
            a = l.weights;
            b = l.delta + i*m*k;
            c = net.workspace;
            if (l.size == 1) {
                c = imd;
            }

            gemm_tn(n,k,m,a,n,b,k,c,k);

            if (l.size != 1) {
                col2im_cpu(net.workspace, l.c, l.h, l.w, l.size, l.stride, l.pad, imd);
            }
        }
    }
}

void update_conv_layer(conv_layer l, int batch, float learning_rate, float momentum, float decay)
{

    axpy(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scale(l.n, momentum, l.bias_updates, 1);

    axpy(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy(l.nweights, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scale(l.nweights, momentum, l.weight_updates, 1);
}

 void test_convolutional_layer()
 {
     conv_layer l = init_convolutional_layer(1, 5, 5, 1, 1, 3, 2, 1);
     float data[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25};
     network state = {0};
     size_t workspace_size = 0;
     if(l.workspace_size > workspace_size) workspace_size = l.workspace_size;
     state.workspace = calloc(1, workspace_size);
     state.input = data;
     conv_fwd(l, state);
 }
