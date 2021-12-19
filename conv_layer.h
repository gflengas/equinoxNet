#ifndef DEVILTEST_CONV_LAYER_H
#define DEVILTEST_CONV_LAYER_H
#include "core.h"

typedef struct{

    int batch;
    int inputs;
    int outputs;
    int nweights;
    int nbiases;
    int h,w,c;
    int out_h, out_w, out_c;
    int n;
    int size;
    int stride;
    int pad;

    float * biases;
    float * bias_updates;

    float * weights;
    float * weight_updates;

    float * delta;//+
    float * output;//+

    size_t workspace_size; //+
}conv_layer;

conv_layer init_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding);
void conv_fwd(const conv_layer layer, network net);
void conv_bwd(conv_layer layer, network net);
void update_conv_layer(conv_layer layer, int batch, float learning_rate, float momentum, float decay);
void fill_cpu(int N, float ALPHA, float *X, int INCX);
void test_convolutional_layer();
#endif //DEVILTEST_CONV_LAYER_H
