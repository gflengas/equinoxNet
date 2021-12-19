#ifndef DEVILTEST_CUSTOM_MATH_H
#define DEVILTEST_CUSTOM_MATH_H
#include "core.h"
#include <math.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TWO_PI 6.2831853071795864769252866f

void normalize_array(float *a, int n);
float variance_array(float *a, int n);
float sum_array(float *a, int n);
float mean_array(float *a, int n);
float rand_normal();

void scale(int N, float ALPHA, float *X, int INCX);
void axpy(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void fill_cpu(int N, float ALPHA, float *X, int INCX);
void gemm_nn(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc);
void gemm_nt(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc);
void gemm_tn(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc);
void gemm_tt(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc);

void im2col_cpu(float* data_im,int channels, int height, int width, int ksize, int stride, int pad, float* data_col);
float im2col_get_pixel(float *im, int height, int width, int row, int col, int channel, int pad);
void col2im_cpu(float* data_col, int channels, int height, int width, int ksize, int stride, int pad, float* data_im);
void col2im_add_pixel(float *im, int height, int width,int row, int col, int channel, int pad, float val);
//activation functions
static inline float relu_activate(float x){return x*(x>0);}
static inline float elu_activate(float x){return (x >= 0)*x + (x < 0)*(exp(x)-1);}
static inline float leaky_activate(float x){return (x>0) ? x : .1*x;}
//activation gradients
static inline float leaky_gradient(float x){return (x>0) ? 1 : .1;}
static inline float relu_gradient(float x){return (x>0);}
static inline float elu_gradient(float x){return (x >= 0) + (x < 0)*(x + 1);}
void top_k(float *a, int n, int k, int *index);
float matrix_topk_accuracy(matrix truth, matrix guess, int k);
float rand_uniform(float min, float max);
#endif //DEVILTEST_CUSTOM_MATH_H
