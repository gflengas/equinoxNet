#ifndef DEVILTEST_CONNECTED_LAYER_H
#define DEVILTEST_CONNECTED_LAYER_H
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
}FC_layer;
FC_layer init_FC_layer(int batch, int inputs, int outputs);

void FC_layer_fwd(FC_layer l, network net);
void FC_layer_bwd(FC_layer l, network net);
void FC_update(FC_layer l, int batch, float learning_rate, float momentum, float decay);
#endif //DEVILTEST_CONNECTED_LAYER_H
