#ifndef HELPERS_H
#define HELPERS_H

#include <string.h>
#include <time.h>
#include <math.h>
#include "array.h"


#define PRINT_ARRAY(arr, len, fmt)                             \
    do {                                                       \
    printf("[");                                                \
    for (int i = 0; i < len; i++) {                             \
        if (i > 0) printf(", ");                                \
        printf(fmt, arr[i]);                                    \
    }                                                             \
    printf("]");                                                  \
    } while(0);                                                    \


#define PI 3.14159265358979323846

typedef struct Function* FunctionPtr;

typedef struct {
    FunctionPtr *data;
    int size;
    int capacity;
} List_FunctionPtr;


bool helper_calc_stride(int64_t **out_stride, uint64_t ndim, uint64_t *shape, size_t dtype_size);

bool helper_calc_backstride(int64_t **out_backstride, uint64_t ndim, int64_t *strides, uint64_t *shape);

bool helper_buffer_of_zeros(void **in_ptr, uint64_t size_of_element, uint64_t number_ele);

bool helper_buffer_of_ones(void **in_ptr, uint64_t size_of_element, uint64_t number_ele, dtype type);

bool helper_buffer_with_value(void **buffer, size_t elesize, uint64_t size, dtype type, double value);

bool helper_buffer_arange(void **buffer, size_t elesize, uint64_t size, dtype type, double start, double step);

bool helper_buffer_linspace(void **buffer, size_t elesize, uint64_t size, dtype type, double start, double stop);

bool helper_buffer_randu(void **buffer, size_t elesize, uint64_t size, dtype type, double low, double high);

bool helper_buffer_randn(void **buffer, size_t elesize, uint64_t size, dtype type, double mean, double stddev);

bool helper_buffer_randint(void **buffer, size_t elesize, uint64_t size, dtype type, int64_t low, int64_t high);

bool list_FunctionPtr_create(List_FunctionPtr **out, int initial_capacity);

bool list_FunctionPtr_push(List_FunctionPtr *list, FunctionPtr value);

bool list_FunctionPtr_get(List_FunctionPtr *list, int index, FunctionPtr *out);

bool list_FunctionPtr_free(List_FunctionPtr *list);




#endif // HELPERS_H