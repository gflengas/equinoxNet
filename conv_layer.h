#ifndef DEVILTEST_CONV_LAYER_H
#define DEVILTEST_CONV_LAYER_H
#include "core.h"

typedef layer conv_layer;

conv_layer init_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding);
void conv_fwd(conv_layer layer, network_state state);
void conv_bwd(conv_layer layer, network_state state);
void update_conv_layer(conv_layer layer, int batch, float learning_rate, float momentum, float decay);
void fill_cpu(int N, float ALPHA, float *X, int INCX);
void test_convolutional_layer();
#endif //DEVILTEST_CONV_LAYER_H
