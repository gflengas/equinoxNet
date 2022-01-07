#ifndef DEVILTEST_MAXPOOL_LAYER_H
#define DEVILTEST_MAXPOOL_LAYER_H

#include "core.h"
#include <float.h>

typedef layer maxpool_layer;

maxpool_layer init_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding);
void maxpool_fwd(const maxpool_layer l, network_state state);
void maxpool_bwd(const maxpool_layer l, network_state state);

#endif
