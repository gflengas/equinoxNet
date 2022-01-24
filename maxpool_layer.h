#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "core.h"
#include <float.h>

typedef struct{

    int batch;
    int inputs;
    int outputs;
    int h,w,c;
    int out_h, out_w, out_c;
    int size;
    int stride;
    int pad;

    int * indexes;

    float * delta;
    float * output;
}maxpool_layer;

maxpool_layer init_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding);
void maxpool_fwd(maxpool_layer l, network net);
void maxpool_bwd(maxpool_layer l, network net);
void free_maxpool_layer(maxpool_layer l);
#endif