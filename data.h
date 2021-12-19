#ifndef DEVILTEST_DATA_H
#define DEVILTEST_DATA_H
#include "core.h"
#include "custom_math.h"
#include <limits.h>

data load_categorical_data_csv(char *filename, int target, int k);
matrix csv_to_matrix(char *filename);
char *fgetl(FILE *fp);
int count_fields(char *line);
float *parse_fields(char *line, int n);
float *pop_column(matrix *m, int c);
float **one_hot_encode(float *a, int n, int k);
void get_random_batch(data d, int n, float *X, float *y);
void get_next_batch(data d, int n, int offset, float *X, float *y);
void normalize_data_rows(data d);
float batch_acc(int batch,int k, float *guess, float *truth);
#endif //DEVILTEST_DATA_H
