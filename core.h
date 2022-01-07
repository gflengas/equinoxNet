#ifndef DEVILTEST_CORE_H
#define DEVILTEST_CORE_H
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

struct network;
typedef struct network network;

struct network_state;
typedef struct network_state network_state;

struct layer;
typedef struct layer layer;

typedef struct matrix{
    int rows, cols;
    float **vals;
} matrix;

typedef struct{
    int w, h;
    matrix X;
    matrix y;
} data;

typedef enum {
    CONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    NETWORK,
    BLANK
} LAYER_TYPE;


struct layer {
    LAYER_TYPE type;

    void (*forward)(struct layer, struct network_state);

    void (*backward)(struct layer, struct network_state);

    void (*update)(struct layer, int, float, float, float);

    int batch;
    int inputs;
    int outputs;
    int nweights;
    int nbiases;
    int h, w, c;
    int out_h, out_w, out_c;
    int n;
    int size;
    int stride;
    int pad;

    float *biases;
    float *bias_updates;

    float *weights;
    float *weight_updates;

    float *delta;
    float *output;

    int * indexes;

    float * cost;
    float * loss;
    size_t workspace_size;
};

typedef struct network{
    int n;
    int batch;
    float epoch;
    float *output;
    layer *layers;

    float learning_rate;
    float momentum;
    float decay;

    int inputs;
    int outputs;
    int truths;
    int h, w, c;

    float *input;
    float *truth;
    float *delta;

    float *workspace;
    float *cost;

} network;

typedef struct network_state{
    float *truth;
    float *input;
    float *delta;
    float *workspace;
    int index;
    network net;
} network_state;

#endif //DEVILTEST_CORE_H
