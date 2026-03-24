#ifndef ARRAYLIST_H
#define ARRAYLIST_H

#include <string.h>
#include <time.h>
#include <math.h>
#include "array.h"
#include "backend.h"
#include "dtypes.h"

typedef struct {
    int noe;
    int capacity;
    array **arrays;
    dtype common_type;
    uint64_t *broadcasted_shape;
    uint64_t broadcasted_ndim;
    bool list_common_type_set;
    bool list_broadcasted;
}ArrayList;


bool arraylist_create(ArrayList **out_list);
bool arraylist_append(ArrayList *list, array *array);
bool arraylist_get_arrays(ArrayList *list, array ***out_arrays);

bool arraylist_broadcast_and_max_ndim(ArrayList *list, uint64_t **out_shape, uint64_t *out_ndim);
bool arraylist_broadcast_stride_from_shape(ArrayList *list, uint64_t *iter_shape, uint64_t iter_ndim, int64_t ***out_strides);
bool arraylist_broadcast_and_max_ndim_blas(ArrayList *list, uint64_t **out_shape, uint64_t *out_ndim, uint64_t len_of_inner_ndim);
bool arraylist_broadcast_for_matmul(ArrayList *list, uint64_t **out_shape, uint64_t *out_ndim);

bool arraylist_find_common_dtype(ArrayList *list, dtype *out_result_type);
bool arraylist_get_common_dtype(ArrayList *list, dtype *out_result_type);
bool arraylist_cast_to_common_dtype(ArrayList *list);
bool arraylist_cast_to_dtype(ArrayList *list, dtype to_type);

bool arraylist_get_common_backend(ArrayList *list, Backend *out_backend);

bool arraylist_remove_last(ArrayList *list);
bool arraylist_free(ArrayList *list);




#endif // ARRAYLIST_H