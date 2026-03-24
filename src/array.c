#include "array.h"
#include <stdlib.h>


bool array_create_array(array **out_array, dtype type, uint64_t ndim, uint64_t *shape, Backend backend) {
    *out_array = malloc(sizeof(array));
    if (*out_array == NULL) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "Failed to allocate memory of size %zu for array", sizeof(array));
        return false;
    }

    (*out_array)->shape = malloc(sizeof(uint64_t) * ndim);
    if ((*out_array)->shape == NULL) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "Failed to allocate memory of size %zu for array shape", sizeof(uint64_t) * ndim);
        return false; 
    }

    size_t dtype_size;
    if (dtypes_sizeof_type(&dtype_size, type) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: dtypes_sizeof_type");
        free_struct(*out_array, STRUCT_ARRAY);
        return false;
    }

    int64_t *stride;
    if (array_calc_stride(&stride, ndim, shape, dtype_size) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: array_calc_stride");
        free_struct(*out_array, STRUCT_ARRAY);
        return false;
    }

    dtypes *out_type;
    if(dtypes_create_dtype(&out_type, type) == false){
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: dtypes_create_dtype");
        free_struct(*out_array, STRUCT_ARRAY);
        return false;
    }

    memcpy((*out_array)->shape, shape, sizeof(uint64_t) * ndim);

    (*out_array)->ndim = ndim;
    (*out_array)->stride = stride;
    (*out_array)->elesize_in_bytes = dtype_size;
    (*out_array)->refcount = 1;
    (*out_array)->type_obj = out_type;
    (*out_array)->owns_data = false;

    if (backend >= BACKEND_COUNT) { 
        REPORT_ERROR(ERR_INPUT_OUT_OF_BOUNDS, "backend is not valid. must be < BACKEND_COUNT");
        free_struct(*out_array, STRUCT_ARRAY);
        return false; 
    }
    (*out_array)->backend = backend;
    (*out_array)->size = 1;

    for (int dim = 0; dim < (*out_array)->ndim; dim++) {
        (*out_array)->size *= shape[dim];
    }

    (*out_array)->data = NULL;


    return true;
}

bool array_add_data(array *array, void *data) {
    if(!array || !data) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Array object or data input is not valid");
        return false;
    }

    if(!array->type_obj) {
        REPORT_ERROR(ERR_STRUCT_FIELD_NOT_INIT, "array type field not initialized");
        return false;
    }

    size_t elesize = array->elesize_in_bytes;

    uint64_t size = array->size; 

    if (array->data) {
        free(array->data);
    }

    array->data = malloc(elesize * size);
    if (!array->data) { 
        REPORT_ERROR(ERR_MALLOC_FAILED, "Failed to allocate memory of size %zu for array data", elesize*size);
        return false;
    }

    memcpy(array->data, data, elesize * size);

    return true;
}

bool array_add_buffer(array *array) {
    if (!array) { 
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Array object is not valid");
        return false; 
    }

    size_t total_bytes = (size_t)(array->elesize_in_bytes * array->size);

    if (array->data) {
        free(array->data);
    }

    array->data = malloc(total_bytes);

    if (!array->data) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "Failed to allocate memory of size %zu for array data", total_bytes);
        return false;
    }

    array->owns_data = true;

    return true;
}

bool array_set_shape(array *arr, uint64_t *shape) {
    if (!arr || !shape) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Array object or shape input is not valid");
        return false;
    }
    arr->shape = shape;
    return true;
}

bool array_set_stride(array *array, int64_t *stride) {
    if (!array || !stride) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Array object or stride input is not valid");
        return false;
    }
    array->stride = stride;
    return true;
}

bool array_set_elesize(array *array, size_t elesize) {
    if (!array || elesize <= 0) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Array object is not valid or elesize input is <= 0");
        return false;
    }
    array->elesize_in_bytes = elesize;
    return true;
}

bool array_get_shape(array *array, uint64_t **out_shape) {
    if (!array || !out_shape) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Array object or shape field is not valid");
        return false;
    }
    *out_shape = array->shape;
    return true;
}

bool array_get_stride(array *array, int64_t **out_stride) {
    if (!array || !out_stride) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Array object or stride field is not valid");
        return false;
    }
    *out_stride = array->stride;
    return true;
}

bool array_get_ndim(array *array, uint64_t *out_ndim) {
    if (!array || !out_ndim) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Array object is not valid or out_ndim input is not a valid ptr");
        return false;
    }
    *out_ndim = array->ndim;
    return true;
}

bool array_get_size(array *array, uint64_t *out_size) {
    if (!array || !out_size) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Array object is not valid or out_size input is not a valid ptr");
        return false;
    }
    *out_size = array->size;
    return true;
}

bool array_get_elesize(array *array, int *out_elesize) {
    if (!array || !out_elesize) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Array object is not valid or out_elesize input is not a valid ptr");
        return false;
    }
    *out_elesize = array->elesize_in_bytes;
    return true;
}

bool array_get_dtype(array *array, dtype *out_dtype) {
    if (!array) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Array object is not valid or out_dtype input is not a valid ptr");
        return false;
    }
    *out_dtype = array->type_obj->type;
    return true;
}

bool array_get_backend(array *array, Backend *out_backend) {
    if (!array || !out_backend) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Array object is not valid or out_backend input is not a valid ptr");
        return false;
    }
    *out_backend = array->backend;
    return true;
}

bool array_cast_to(array **out_array, array *in_array, dtype cast_to) {
    if (!in_array || !in_array->data) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "input array object or data field is not valid");
        return false;
    }

    uint64_t ndim = in_array->ndim;
    uint64_t *shape = in_array->shape;
    Backend backend = in_array->backend;
    
    size_t new_elesize;
    if (!dtypes_sizeof_type(&new_elesize, cast_to)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "dtypes_sizeof_type");
        return false;
    }

    if (!array_create_array(out_array, cast_to, ndim, shape, backend)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_create_array");
        return false;
    }

    void *new_data = malloc(in_array->size * new_elesize);
    if (!new_data) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "Failed to allocate memory of size %zu for array new_data", in_array->size * new_elesize);
        return false;
    }

    dtype current_type = in_array->type_obj->type;
    if (!dtypes_fast_contiguous_cast(in_array->data, current_type, new_data, cast_to, in_array->size)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "dtypes_fast_contiguous_cast");
        return false;
    }

    (*out_array)->data = new_data;
    (*out_array)->size = in_array->size;
    (*out_array)->owns_data = true; 

    return true;
}

bool array_inc_refcount(array *array) {
    if (!array) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Array object is not valid");
        return false;
    }
    
    array->refcount++;
    return true;
}

bool array_dec_refcount(array *array) {
    if (!array) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Array object is not valid");
        return false;
    }

    array->refcount--;

    if (array->refcount == 0) {
        array_free_array(&array); 
    }

    return true;
}

bool array_calc_stride(int64_t **out_stride, uint64_t ndim, uint64_t *shape, size_t dtype_size) {

    *out_stride = malloc(ndim * sizeof(int64_t));

    if (out_stride == NULL) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "Failed to allocate memory of size %zu for out_stride", ndim * sizeof(int64_t));
        return false;
    }

    (*out_stride)[ndim - 1] = dtype_size;

    for (int64_t i = ndim - 2; i >= 0; i--) {
        (*out_stride)[i] = (*out_stride)[i + 1] * shape[i + 1];
    }

    return true;
}

bool array_free_array(array **array) {
    if(!array) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Array **object is not valid ptr");
        return false;
    }
    
    if ((*array)->type_obj) {
        bool code = dtypes_free_dtype((*array)->type_obj);
        if (code == false) {

        }
    }
    free((*array)->shape);
    free((*array)->stride);
    free((*array)->data);
    free((*array));
    *array = NULL;
    return true;
}

