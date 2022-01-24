#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H
#include "core.h"
#include "custom_math.h"

typedef struct{

    int batch;
    int inputs;
    int outputs;

    float * cost;
    float * loss;

    float * delta;
    float * output;
}softmax_layer;

softmax_layer init_softmax_layer(int batch, int inputs);
void softmax_fwd(softmax_layer l, network net);
void softmax(float *input, int n, float *output);
void softmax_cros_ent(int n, float const *pred, float const *truth, float *delta, float *error);
void softmax_bwd(softmax_layer l, network net);
void free_softmax_layer(softmax_layer l);
#endif
