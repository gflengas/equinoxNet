#ifndef DEVILTEST_SOFTMAX_LAYER_H
#define DEVILTEST_SOFTMAX_LAYER_H
#include "core.h"
#include "custom_math.h"

typedef layer softmax_layer;

softmax_layer init_softmax_layer(int batch, int inputs);
void softmax_fwd(const softmax_layer l, network_state state);
void softmax(float *input, int n, float *output);
void softmax_cros_ent(int n, float *pred, float *truth, float *delta, float *error);
void softmax_bwd(const softmax_layer l, network_state state);
#endif //DEVILTEST_SOFTMAX_LAYER_H
