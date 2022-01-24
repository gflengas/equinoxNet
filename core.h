#ifndef CORE_H
#define CORE_H
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct{
    int batch;
    float learning_rate;
    float momentum;
    float decay;
    int adam;
    float B1;
    float B2;
    float eps;
    int t;
} update_args;

typedef struct matrix{
    int rows, cols;
    float **vals;
} matrix;

typedef struct{
    int w, h;
    matrix X;
    matrix y;
} data;


typedef struct network{
    int n;
    int batch;
    float epoch;
    float *output;

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


#endif
