#include "custom_math.h"

void normalize_array(float *a, int n) {
    int i;
    float mu = mean_array(a, n); //mean
    float sigma = sqrtf(variance_array(a, n));
    for (i = 0; i < n; ++i) {
        a[i] = (a[i] - mu) / sigma;
    }
    mu = mean_array(a, n);
    sigma = sqrt(variance_array(a, n));
}

float variance_array(float *a, int n) {
    int i;
    float sum = 0;
    float mean = mean_array(a, n);
    for (i = 0; i < n; ++i) sum += (a[i] - mean) * (a[i] - mean);
    float variance = sum / n;
    return variance;
}

float sum_array(float *a, int n) {
    int i;
    float sum = 0;
    for (i = 0; i < n; ++i) sum += a[i];
    return sum;
}

float mean_array(float *a, int n) {
    return sum_array(a, n) / n;
}

// From http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
float rand_normal() {
    static int haveSpare = 0;
    static double rand1, rand2;

    if (haveSpare) {
        haveSpare = 0;
        return sqrt(rand1) * sin(rand2);
    }

    haveSpare = 1;

    rand1 = rand() / ((double) RAND_MAX);
    if (rand1 < 1e-100) rand1 = 1e-100;
    rand1 = -2 * log(rand1);
    rand2 = (rand() / ((double) RAND_MAX)) * TWO_PI;

    return sqrt(rand1) * cos(rand2);
}

//a combination of scalar multiplication and vector addition. We want to parallelize the code using the task directive in order to express parallelism.
void axpy(int N, float ALPHA, float *X, int INCX, float *Y, int INCY) {
    int i;
    for (i = 0; i < N; ++i) Y[i * INCY] += ALPHA * X[i * INCX];
}

void scale(int N, float ALPHA, float *X, int INCX) {
    int i;
    for (i = 0; i < N; ++i) X[i * INCX] *= ALPHA;
}

void gemm_nn(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float A_PART = A[i * lda + k];
            for (int j = 0; j < N; ++j) {
                C[i * ldc + j] += A_PART * B[k * ldb + j];

            }
        }
    }
}

void gemm_nt(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < K; ++k) {
                C[i * ldc + j] += A[i * lda + k] * B[j * ldb + k];
            }
        }
    }
}

void gemm_tn(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            for (int j = 0; j < N; ++j) {
                C[i * ldc + j] += A[k * lda + i] * B[k * ldb + j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < K; ++k) {
                C[i * ldc + j] += A[i + k * lda] * B[k + j * ldb];
            }
        }
    }
}

float im2col_get_pixel(float *im, int height, int width, int row, int col, int channel, int pad) {
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width)
        return 0;
    return im[col + width * (row + height * channel)];
}


void im2col_cpu(float *data_im,
                int channels, int height, int width,
                int ksize, int stride, int pad, float *data_col) {
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (int h = 0; h < height_col; ++h) {
            for (int w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width,
                                                       im_row, im_col, c_im, pad);
            }
        }
    }
}

void col2im_add_pixel(float *im, int height, int width,
                      int row, int col, int channel, int pad, float val) {
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width)
        return;
    im[col + width * (row + height * channel)] += val;
}

void col2im_cpu(float *data_col,
                int channels, int height, int width,
                int ksize, int stride, int pad, float *data_im) {
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (int h = 0; h < height_col; ++h) {
            for (int w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                float val = data_col[col_index];
                col2im_add_pixel(data_im, height, width,
                                 im_row, im_col, c_im, pad, val);
            }
        }
    }
}

void fill(int N, float ALPHA, float *X, int INCX) {
    int i;
    for (i = 0; i < N; ++i) X[i * INCX] = ALPHA;
}

float rand_uniform(float min, float max) {
    if (max < min) {
        float swap = min;
        min = max;
        max = swap;
    }

#if (RAND_MAX < 65536)
    int rnd = rand() * (RAND_MAX + 1) + rand();
    return ((float) rnd / (RAND_MAX * RAND_MAX) * (max - min)) + min;
#else
    return ((float)rand() / RAND_MAX * (max - min)) + min;
#endif
    //return (random_float() * (max - min)) + min;
}