#ifndef CORE_H
#define CORE_H
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

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
    int batch;

    float learning_rate;
    float momentum;
    float decay;

    int inputs;
    int truths;
    int h, w, c;

    float *input;
    float *truth;
    float *delta;

    float *cost;

} network;


#endif
