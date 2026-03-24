


#ifndef ARRAY_H
#define ARRAY_H

#include <stdbool.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>
#include <stdio.h>
#include "dtypes.h"
#include "Backend.h"
#include "Error.h"
// #include "helpers.h"
#include "mem.h"


typedef struct array {
    dtypes *type_obj;
    Backend backend;
    uint64_t *shape;
    int64_t *stride;
    uint64_t size;
    uint64_t ndim;
    int64_t refcount;
    size_t elesize_in_bytes;
    bool owns_data;
    char *data;
} array;



bool array_create_array(array **out_array, dtype type, uint64_t ndim, uint64_t *shape, Backend backend);
bool array_add_data(array *array, void *data);
bool array_add_buffer(array *array);

bool array_set_shape(array *array, uint64_t *shape);
bool array_set_stride(array *array, int64_t *stride);
bool array_set_ndim(array *array, uint64_t ndim);
bool array_set_size(array *array, uint64_t size);
bool array_set_elesize(array *array, size_t elesize);


bool array_get_shape(array *array, uint64_t **out_shape);
bool array_get_stride(array *array, int64_t **out_stride);
bool array_get_ndim(array *array, uint64_t *out_ndim);
bool array_get_size(array *array, uint64_t *out_size);
bool array_get_elesize(array *array, int *out_elesize);
bool array_get_dtype(array *array, dtype *out_dtype);
bool array_get_backend(array *array, Backend *out_backend);

bool array_cast_to(array **out_array, array *in_array, dtype cast_to);

bool array_inc_refcount(array *array);
bool array_dec_refcount(array *array);


bool array_free_array(array **array);

bool array_calc_stride(int64_t **out_stride, uint64_t ndim, uint64_t *shape, size_t dtype_size);





#endif // ARRAY_H