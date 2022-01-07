#ifndef DEVILTEST_CONNECTED_LAYER_H
#define DEVILTEST_CONNECTED_LAYER_H
#include "core.h"

typedef layer FC_layer;
FC_layer init_FC_layer(int batch, int inputs, int outputs);

void FC_layer_fwd(FC_layer l, network_state state);
void FC_layer_bwd(FC_layer l, network_state state);
void FC_update(FC_layer l, int batch, float learning_rate, float momentum, float decay);
#endif //DEVILTEST_CONNECTED_LAYER_H
