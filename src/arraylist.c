#include "arraylist.h"

uint64_t *broadcast_output_matmul(uint64_t *s1, uint64_t *s2, uint64_t nds1, uint64_t nds2, uint64_t *out_ndim) {
    uint64_t *sh1 = s1;
    uint64_t *sh2 = s2;
    uint64_t dim1 = nds1;
    uint64_t dim2 = nds2;

    int sh1_alloc = 0;
    int sh2_alloc = 0;

    int left_was_1d = 0;
    int right_was_1d = 0;

    if (nds1 == 1) {
        sh1 = malloc(2 * sizeof(uint64_t));
        if (!sh1) return NULL;
        sh1[0] = 1;
        sh1[1] = s1[0];
        dim1 = 2;
        sh1_alloc = 1;
        left_was_1d = 1;
    }

    if (nds2 == 1) {
        sh2 = malloc(2 * sizeof(uint64_t));
        if (!sh2) {
            if (sh1_alloc) free(sh1);
            return NULL;
        }
        sh2[0] = s2[0];
        sh2[1] = 1;
        dim2 = 2;
        sh2_alloc = 1;
        right_was_1d = 1;
    }

    uint64_t max_ndim = (dim1 > dim2) ? dim1 : dim2;

    uint64_t *p1 = sh1;
    uint64_t *p2 = sh2;

    int p1_alloc = 0;
    int p2_alloc = 0;

    if (dim1 < max_ndim) {
        p1 = malloc(max_ndim * sizeof(uint64_t));
        if (!p1) goto fail;
        p1_alloc = 1;

        uint64_t offset = max_ndim - dim1;
        for (uint64_t i = 0; i < max_ndim; i++) {
            p1[i] = (i < offset) ? 1 : sh1[i - offset];
        }
    }

    if (dim2 < max_ndim) {
        p2 = malloc(max_ndim * sizeof(uint64_t));
        if (!p2) goto fail;
        p2_alloc = 1;

        uint64_t offset = max_ndim - dim2;
        for (uint64_t i = 0; i < max_ndim; i++) {
            p2[i] = (i < offset) ? 1 : sh2[i - offset];
        }
    }

    uint64_t *shape = malloc(max_ndim * sizeof(uint64_t));
    if (!shape) goto fail;

    for (uint64_t dim = 0; dim < max_ndim - 2; dim++) {
        uint64_t d1 = p1[dim];
        uint64_t d2 = p2[dim];

        if (d1 == d2) {
            shape[dim] = d1;
        } else if (d1 == 1) {
            shape[dim] = d2;
        } else if (d2 == 1) {
            shape[dim] = d1;
        } else {
            free(shape);
            goto fail;
        }
    }

    if (p1[max_ndim - 1] != p2[max_ndim - 2]) {
        free(shape);
        goto fail;
    }

    shape[max_ndim - 2] = p1[max_ndim - 2];
    shape[max_ndim - 1] = p2[max_ndim - 1];

    uint64_t final_ndim = max_ndim;

    if (left_was_1d) {
        for (uint64_t i = max_ndim - 2; i < max_ndim - 1; i++) {
            shape[i] = shape[i + 1];
        }
        final_ndim--;
    }

    if (right_was_1d) {
        final_ndim--;
    }

    uint64_t *final_shape = malloc(final_ndim * sizeof(uint64_t));
    if (!final_shape) {
        free(shape);
        goto fail;
    }

    for (uint64_t i = 0; i < final_ndim; i++) {
        final_shape[i] = shape[i];
    }

    free(shape);

    if (sh1_alloc) free(sh1);
    if (sh2_alloc) free(sh2);
    if (p1_alloc) free(p1);
    if (p2_alloc) free(p2);

    *out_ndim = final_ndim;
    return final_shape;

fail:
    if (sh1_alloc) free(sh1);
    if (sh2_alloc) free(sh2);
    if (p1_alloc) free(p1);
    if (p2_alloc) free(p2);
    return NULL;
}


bool arraylist_create(ArrayList **out_list) {
    if (out_list == NULL) return false;

    ArrayList *list = malloc(sizeof(ArrayList));
    if (!list) return false;

    list->noe = 0;
    list->capacity = 4;
    list->common_type = DTYPE_COUNT;
    list->list_common_type_set = false;

    list->arrays = malloc(list->capacity * sizeof(array*));

    if (!list->arrays) {
        free(list);
        return false;
    }

    *out_list = list;
    return true;
}

bool arraylist_append(ArrayList *list, array *arr) {
    if (!list || !arr) {  return false; }

    if (list->noe >= list->capacity) {
        int new_capacity = list->capacity * 2;
        array **new_ptr = realloc(list->arrays, new_capacity * sizeof(array*));
        
        if (!new_ptr) {  return false; }

        list->arrays = new_ptr;
        list->capacity = new_capacity;
    }

    list->arrays[list->noe] = arr;
    list->noe++;
    return true;
}

bool arraylist_get_arrays(ArrayList *list, array ***out_arrays) {
    if (!list || !out_arrays) return false;
    *out_arrays = list->arrays;
    return true;
}

bool arraylist_broadcast_and_max_ndim(ArrayList *list, uint64_t **out_shape, uint64_t *out_ndim) {

    if (!list) return false;

    uint64_t ndim = 0;

    for (int i = 0; i < list->noe; i++) { ndim =  ndim > list->arrays[i]->ndim ? ndim : list->arrays[i]->ndim; }

    uint64_t *shape =  (uint64_t *) malloc(ndim * sizeof(uint64_t)); 
    if (!shape) return false;

    for (int i = 0; i < ndim; i++) shape[i] = 1;

    for (int arr = 0; arr < list->noe; arr++) {
        const array *ar = list->arrays[arr];

        for (int dim = 0; dim < ndim; dim++) {

            int idx = dim - (ndim - ar->ndim); // x = (ndim - op->ndim) skips leading dimension that are not aligned with op, 
                                                // dim - x checks if op has the corresponding dimension in the broadcast shape (src > 0)
            uint64_t dim_size = (idx >= 0) ? (ar->shape)[idx] : 1; // If it has the dimension we use it, else we use 1

            if (shape[dim] != 1 && dim_size != 1 && shape[dim] != dim_size) {
                free(shape);
                return false; // Shapes are not compatiable
            }

            if (shape[dim] == 1) shape[dim] = dim_size; // Only update dimensions waiting to be updated

        }
    }

    *out_shape = shape;

    *out_ndim = ndim;

    list->broadcasted_shape = shape;
    list->broadcasted_ndim = ndim;

    list->list_broadcasted = true;

    return true;

}

bool arraylist_broadcast_for_matmul(ArrayList *list, uint64_t **out_shape, uint64_t *out_ndim) {
    
    if (!list || list->noe == 0) return false;

    uint64_t *current_shape = NULL;
    uint64_t current_ndim = 0;

    for (uint64_t i = 0; i < list->noe; i++) {

        array *ar = list->arrays[i];
        if (!ar || ar->ndim == 0) return false;

        if (i == 0) {
            current_ndim = ar->ndim;
            current_shape = malloc(current_ndim * sizeof(uint64_t));
            if (!current_shape) return false;

            for (uint64_t j = 0; j < current_ndim; j++) {
                current_shape[j] = ar->shape[j];
            }
        } else {
            uint64_t new_ndim = 0;
            uint64_t *new_shape = broadcast_output_matmul(current_shape, ar->shape, current_ndim, ar->ndim, &new_ndim);

            free(current_shape);

            if (!new_shape) return false;

            current_shape = new_shape;
            current_ndim = new_ndim;
        }
    }

    *out_shape = current_shape;
    *out_ndim = current_ndim;

    return true;
}


bool arraylist_broadcast_stride_from_shape(ArrayList *list, uint64_t *iter_shape, uint64_t iter_ndim, int64_t ***out_strides) {

    // Here we're setting the strides of broadcasted dimensions to zero
    if (!list) return false;

    if (!iter_shape) return false;

    if (!out_strides) return false;

    int64_t **result_strides = malloc(sizeof(int64_t *) * list->noe);
    if (!result_strides) return false;

    for (int arr = 0; arr < list->noe; arr++) {

        const array *ar = list->arrays[arr];
        int64_t *stride = (int64_t *) malloc(iter_ndim * sizeof(int64_t));

        if (!stride) {
            for (int i = 0; i < arr; i++) {
                free(result_strides[i]);
            }
            free(result_strides);
            return false;
        }

        for (int dim = 0; dim < iter_ndim; dim++) {
            stride[dim] = 0;
        }

        int64_t lead = iter_ndim - ar->ndim; // So we can get to the dimension where the specific arrays strides begin

        for (int dim = 0; dim < ar->ndim; dim++) {
            int64_t str = ar->stride[dim];
            int64_t broadcast_dim = lead + dim;
            stride[broadcast_dim] = (iter_shape[broadcast_dim] == 1) ? 0 : str; // Dimensions of 1 have a stride of 0
        }

        result_strides[arr] = stride;
    }

    *out_strides = result_strides;

    return true;
}

bool arraylist_find_common_dtype(ArrayList *list, dtype *out_result_type) {

    if (!list) return false;

    if (!list->arrays) { return false; }

    dtype r_type;

    if (array_get_dtype(list->arrays[0], &r_type) == false) { return false; }

    int is_signed_1;

    int is_signed_2;
    
    for (int i = 1; i < list->noe; i++) {
        
        dtype next_type;

        if (array_get_dtype(list->arrays[i], &next_type) == false) { return false; }

        if (dtypes_dtype_is_signed(r_type, &is_signed_1) == false) { return false; }

        if (dtypes_dtype_is_signed(next_type, &is_signed_2) == false) { return false; }

        if (dtypes_promote_dtype(&r_type, r_type, next_type, is_signed_1, is_signed_2) == false) { return false; }

    }

    list->list_common_type_set = true;

    list->common_type = r_type;

    *out_result_type = r_type;

    return true;
}

bool arraylist_get_common_dtype(ArrayList *list, dtype *out_result_type) {

    if (!list) return false;

    if (list->list_common_type_set) {

        *out_result_type = list->common_type;

        return true;

    } else {

        if (arraylist_find_common_dtype(list, out_result_type) == false) {
            return false;
        }

    }

    *out_result_type = list->common_type;

    return true;



}

bool arraylist_cast_to_common_dtype(ArrayList *list) {
    if (!list || list->noe == 0) { return false; }

    dtype common_type;
    if (!arraylist_get_common_dtype(list, &common_type)) return false;

    for (int i = 0; i < list->noe; i++) {
        array *original_arr = list->arrays[i];
        
        dtype current_type;
        array_get_dtype(original_arr, &current_type); 

        if (current_type == common_type) continue;

        array *casted_arr = NULL;
        if (!array_cast_to(&casted_arr, original_arr, common_type) == false) {
            // Logically, you might want to cleanup previously created 
            // casted arrays here if one fails mid-way.
            return false;
        }

        list->arrays[i] = casted_arr;

    }

    list->list_common_type_set = true;
    list->common_type = common_type;

    return true;
}

bool arraylist_cast_to_dtype(ArrayList *list, dtype to_type) {
    if (!list || list->noe == 0) { return false; }

    for (int i = 0; i < list->noe; i++) {
        array *old_arr = list->arrays[i];
        
        if (old_arr->type_obj->type == to_type) {
            continue;
        }

        array *new_arr = NULL;
        
        if (array_cast_to(&new_arr, old_arr, to_type) == false) {
            return false;
        }

        list->arrays[i] = new_arr;

    }

    return true;
}

bool arraylist_get_common_backend(ArrayList *list, Backend *out_Backend) {
    if (!list || list->noe == 0 || !out_Backend) {
        return false;
    }

    int counts[BACKEND_COUNT] = {0};

    for (int i = 0; i < list->noe; i++) {
        Backend b;
        if (array_get_backend(list->arrays[i], &b)) {
            if (b < BACKEND_COUNT) {
                counts[b]++;
            }
        } else {
            return false;
        }
    }

    int max_count = -1;
    Backend winner = 0; 

    for (int i = 0; i < BACKEND_COUNT; i++) {
        if (counts[i] > max_count) {
            max_count = counts[i];
            winner = (Backend)i;
        }
    }

    *out_Backend = winner;
    return true;
}

bool arraylist_remove_last(ArrayList *list) {
    if (!list) { return false; }

    if (!list->arrays) { return false; }

    if (list->noe <= 0) {return false; }

    list->noe--;
    list->arrays[list->noe] = NULL;

    return true;

}

bool arraylist_free(ArrayList *list) {
    if (list) {
        if (list->arrays) {
            free(list->arrays);
        }
        free(list);
        return true;
    }

    return false;
}
