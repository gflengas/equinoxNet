#ifndef DATA_H
#define DATA_H
#include "core.h"
#include "custom_math.h"
#include <limits.h>

data load_categorical_data_csv(char *filename, int target, int k);
matrix csv_to_matrix(char *filename);
char *fgetl(FILE *fp);
int count_fields(char *line);
float *parse_fields(char *line, int n);
float *pop_column(matrix *m, int c);
float **one_hot_encode(float const *a, int n, int k);
void get_random_batch(data d, int n, float *X, float *y);
void get_next_batch(data d, int n, int offset, float *X, float *y);
void normalize_data_rows(data d);
void free_data(data d);
float batch_acc(int batch,int k, float const *guess, float const *truth);
void free_matrix(matrix m);
#endif 