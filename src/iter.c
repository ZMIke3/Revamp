#include "Iter.h"
#include "iter_functions.h"

// void Iter_print(Iter *iter) {
//     if (!iter) {
//         printf("Iter is NULL\n");
//         return;
//     }

//     printf("=== Iter ===\n");
//     printf("nops:             %d\n",  iter->nops);
//     printf("ndim:             %llu\n", (unsigned long long)iter->ndim);
//     printf("counter:          %llu\n", (unsigned long long)iter->counter);
//     printf("total_size:       %llu\n", (unsigned long long)iter->total_size);
//     printf("inner_loop_size:  %llu\n", (unsigned long long)iter->inner_loop_size);
//     printf("outer_ndim:       %llu\n", (unsigned long long)iter->outer_ndim);
//     printf("inner_ndim:       %llu\n", (unsigned long long)iter->inner_ndim);
//     printf("rndims:           %llu\n", (unsigned long long)iter->rndims);
//     printf("keepdims:         %s\n",   iter->keepdims ? "true" : "false");
//     printf("flag:             %d\n",   (int)iter->flag);

//     printf("arr_a_row_mm:     %llu\n", (unsigned long long)iter->arr_a_row_mm);
//     printf("arr_a_col_mm:     %llu\n", (unsigned long long)iter->arr_a_col_mm);
//     printf("arr_b_row_mm:     %llu\n", (unsigned long long)iter->arr_b_row_mm);
//     printf("arr_b_col_mm:     %llu\n", (unsigned long long)iter->arr_b_col_mm);

//     // shape
//     if (!iter->shape) {
//         printf("shape:            NULL\n");
//     } else {
//         printf("shape:            [");
//         for (uint64_t i = 0; i < iter->ndim; i++)
//             printf("%llu%s", (unsigned long long)iter->shape[i], i < iter->ndim - 1 ? ", " : "");
//         printf("]\n");
//     }

//     // coordinates
//     if (!iter->coordinates) {
//         printf("coordinates:      NULL\n");
//     } else {
//         printf("coordinates:      [");
//         for (uint64_t i = 0; i < iter->outer_ndim; i++)
//             printf("%llu%s", (unsigned long long)iter->coordinates[i], i < iter->outer_ndim - 1 ? ", " : "");
//         printf("]\n");
//     }

//     // baseptrs / dataptrs
//     if (!iter->baseptrs) {
//         printf("baseptrs:         NULL\n");
//     } else {
//         for (int i = 0; i < iter->nops; i++)
//             printf("baseptrs[%d]:      %p\n", i, (void*)iter->baseptrs[i]);
//     }

//     if (!iter->dataptrs) {
//         printf("dataptrs:         NULL\n");
//     } else {
//         for (int i = 0; i < iter->nops; i++)
//             printf("dataptrs[%d]:      %p\n", i, (void*)iter->dataptrs[i]);
//     }

//     // strides [nops][ndim]
//     if (!iter->strides) {
//         printf("strides:          NULL\n");
//     } else {
//         for (int i = 0; i < iter->nops; i++) {
//             if (!iter->strides[i]) {
//                 printf("strides[%d]:       NULL\n", i);
//             } else {
//                 printf("strides[%d]:       [", i);
//                 for (uint64_t d = 0; d < iter->outer_ndim; d++)
//                     printf("%lld%s", (long long)iter->strides[i][d], d < iter->outer_ndim - 1 ? ", " : "");
//                 printf("]\n");
//             }
//         }
//     }

//     // backstrides [nops][ndim]
//     if (!iter->backstrides) {
//         printf("backstrides:      NULL\n");
//     } else {
//         for (int i = 0; i < iter->nops; i++) {
//             if (!iter->backstrides[i]) {
//                 printf("backstrides[%d]:   NULL\n", i);
//             } else {
//                 printf("backstrides[%d]:   [", i);
//                 for (uint64_t d = 0; d < iter->outer_ndim; d++)
//                     printf("%lld%s", (long long)iter->backstrides[i][d], d < iter->outer_ndim - 1 ? ", " : "");
//                 printf("]\n");
//             }
//         }
//     }

//     // inner_strides [nops]
//     if (!iter->inner_strides) {
//         printf("inner_strides:    NULL\n");
//     } else {
//         printf("inner_strides:    [");
//         for (int i = 0; i < iter->nops; i++)
//             printf("%lld%s", (long long)iter->inner_strides[i], i < iter->nops - 1 ? ", " : "");
//         printf("]\n");
//     }

//     // reduce fields
//     if (!iter->_reduce_dims)  printf("_reduce_dims:     NULL\n");
//     if (!iter->reduce_shape)  printf("reduce_shape:     NULL\n");
//     if (!iter->reduce_strides)printf("reduce_strides:   NULL\n");
//     if (!iter->flat_strides)  printf("flat_strides:     NULL\n");

//     printf("============\n");
// }

// void iter_debug_print(Iter *iter) {
//     printf("===== ITER DEBUG START =====\n");

//     printf("nops: %ld\n", iter->nops);
//     printf("ndim: %lu\n", iter->ndim);
//     printf("outer_ndim: %lu\n", iter->outer_ndim);
//     printf("inner_ndim: %lu\n", iter->inner_ndim);
//     printf("total_size: %lu\n", iter->total_size);
//     printf("inner_loop_size: %lu\n", iter->inner_loop_size);
//     printf("counter: %lu\n", iter->counter);

//     // Shape
//     printf("shape: [");
//     for (uint64_t i = 0; i < iter->ndim; i++) {
//         printf("%lu", iter->shape[i]);
//         if (i < iter->ndim - 1) printf(", ");
//     }
//     printf("]\n");

//     // Coordinates
//     printf("coordinates: [");
//     for (uint64_t i = 0; i < iter->ndim; i++) {
//         printf("%lu", iter->coordinates[i]);
//         if (i < iter->ndim - 1) printf(", ");
//     }
//     printf("]\n");

//     // Strides
//     printf("strides:\n");
//     for (int op = 0; op < iter->nops; op++) {
//         printf("  op %d: [", op);
//         for (uint64_t dim = 0; dim < iter->ndim; dim++) {
//             printf("%ld", iter->strides[op][dim]);
//             if (dim < iter->ndim - 1) printf(", ");
//         }
//         printf("]\n");
//     }

//     // Backstrides
//     printf("backstrides:\n");
//     for (int op = 0; op < iter->nops; op++) {
//         printf("  op %d: [", op);
//         for (uint64_t dim = 0; dim < iter->ndim; dim++) {
//             printf("%ld", iter->backstrides[op][dim]);
//             if (dim < iter->ndim - 1) printf(", ");
//         }
//         printf("]\n");
//     }

//     // Inner strides
//     printf("inner_strides: [");
//     for (int i = 0; i < iter->nops; i++) {
//         printf("%ld", iter->inner_strides[i]);
//         if (i < iter->nops - 1) printf(", ");
//     }
//     printf("]\n");

//     // Baseptrs/dataptrs
//     printf("baseptrs / dataptrs:\n");
//     for (int i = 0; i < iter->nops; i++) {
//         printf("  op %d: base=%p data=%p\n", i, iter->baseptrs[i], iter->dataptrs[i]);
//     }

//     // Reduce-related fields
//     printf("_reduce_dims: %p\n", iter->_reduce_dims);
//     printf("keepdims: %p\n", iter->keepdims);
//     printf("reduce_shape: %p\n", iter->reduce_shape);
//     printf("reduce_strides: %p\n", iter->reduce_strides);

//     printf("===== ITER DEBUG END =====\n");
// }

bool Iter_create_iterator(Iter **iter, ArrayList *list, IterCtx *ctx, IterFlags flag) {
    if (!list) {
        printf("List is false\n"); 
        return false; 
    }

    switch (flag) {
        case ITER_ELEMENTWISE:
            if (element_wise_iter_create(iter, list, ctx) == false) {
                REPORT_ERROR(ERR_FUNC_CALL_FAILED, "element_wise_iter_create");
                return false;
            }
            return true;
        break;

        case ITER_REDUCTION:
            if (reduction_iter_create(iter, list, ctx) == false) {
                REPORT_ERROR(ERR_FUNC_CALL_FAILED, "reduction_iter_create");
                return false;
            }
            return true;


        case ITER_BATCHED_MATMUL:
            if (matmul_iter_create(iter, list, ctx) == false) {
                REPORT_ERROR(ERR_FUNC_CALL_FAILED, "matmul_iter_create");
                return false;
            }
            return true;
        break;
    }
}

// static bool Iter_create_elementwise_iterator(Iter **iter, ArrayList *list, IterCtx *ctx, IterFlags flag) {
//     *iter = malloc(sizeof(Iter));
//     if (*iter == NULL) { WRITE("Iter == NULL after malloc\n")return false; }

//     uint64_t *iter_shape;
//     uint64_t iter_ndim;

//     int64_t **iter_strides;
//     int64_t **iter_backstrides;
//     int64_t *iter_inner_strides;
//     uint64_t iter_inner_loop_size = 1;
//     uint64_t iter_total_size = 1;

//     uint64_t *iter_coordinates;

//     uint64_t outer_ndim;
//     uint64_t inner_ndim;
    
//     int64_t nops = list->noe;

//    // WRITE("0.I\n")
//     (*iter)->baseptrs = malloc(sizeof(char*) * nops);
//     (*iter)->dataptrs = malloc(sizeof(char*) * nops);
//     // WRITE("1.I\n")
//     if (!(*iter)->baseptrs || !(*iter)->dataptrs) {WRITE("Iter data and base ptrs == NULL after malloc\n")  return false; }

//     for (int i = 0; i < nops; i++) {
//         (*iter)->baseptrs[i] = list->arrays[i]->data;
//         (*iter)->dataptrs[i] = list->arrays[i]->data;
//     }
//     // WRITE("2.I\n")
//     if (arraylist_broadcast_and_max_ndim(list, &iter_shape, &iter_ndim) == false) { return false; }
//     if (arraylist_broadcast_stride_from_shape(list, iter_shape, iter_ndim, &iter_strides) == false) { return false; }

//     // WRITE("2.0.I\n")
//     iter_backstrides = malloc(sizeof(int64_t *) * nops);
//     if (!iter_backstrides) { return false; }
//     // WRITE("2.1.I\n")
//     iter_coordinates = calloc(sizeof(uint64_t), iter_ndim);
//     if (!iter_coordinates) { return false; }

//     //WRITE("3.I\n")
        
//     for (int ar = 0; ar < nops; ar++) {
         
//         iter_backstrides[ar] = malloc(sizeof(int64_t) * iter_ndim);
//         if (!iter_backstrides[ar]) { return false; }

//         for (uint64_t dim = 0; dim < iter_ndim; dim++) {

//             int64_t backstride = iter_strides[ar][dim] * ((int64_t)iter_shape[dim] - 1);

//             iter_backstrides[ar][dim] = backstride;
//         }

//     }

//     // WRITE("4.I\n")
//         // this might be a bit redundant
//     for (int op = 0; op < nops; op++) {
//         int64_t offset = iter_ndim - list->arrays[op]->ndim;
//         for (int i = 0; i < iter_ndim; i++) {
//             int64_t arr_dim = i - offset;
//             if (arr_dim >= 0) {
//                 if (list->arrays[op]->shape[arr_dim] == 1 && iter_shape[i] > 1) {
//                     iter_strides[op][i] = 0;
//                     iter_backstrides[op][i] = 0;
//                 }
//             } else {
//                 iter_strides[op][i] = 0;
//                 iter_backstrides[op][i] = 0;
//             }
//         }
//     }
       
//     iter_inner_strides = malloc(sizeof(int64_t) * nops);

//     for (int i = 0; i < nops; i++) {
//         iter_inner_strides[i] = iter_strides[i][iter_ndim - 1];
//     }

//     for (uint64_t dim = 0; dim < iter_ndim; dim++) {
//         iter_total_size *= iter_shape[dim]; 
//     }


//     for (int i = ctx->len_of_inner_ndim; i > 0; i--) {
//         iter_inner_loop_size *= iter_shape[iter_ndim - i];
//     }
        
//     outer_ndim = iter_ndim - ctx->len_of_inner_ndim;
//     inner_ndim = ctx->len_of_inner_ndim;
//     iter_total_size = iter_total_size / iter_inner_loop_size;

//     (*iter)->nops = nops;
//     (*iter)->ndim = iter_ndim;
//     (*iter)->shape = iter_shape;

//     (*iter)->strides = iter_strides;
//     (*iter)->backstrides = iter_backstrides;
//     (*iter)->coordinates = iter_coordinates;

//     (*iter)->total_size = iter_total_size;
//     (*iter)->inner_loop_size = iter_inner_loop_size;
//     (*iter)->inner_strides = iter_inner_strides;

//     (*iter)->arr_a_col_mm = ctx->arr_a_col_mm;
//     (*iter)->arr_a_row_mm = ctx->arr_a_row_mm;
//     (*iter)->arr_b_col_mm = ctx->arr_b_col_mm;
//     (*iter)->arr_b_row_mm = ctx->arr_b_row_mm;

//     (*iter)->outer_ndim = outer_ndim;
//     (*iter)->inner_ndim = inner_ndim;
//     (*iter)->counter = 0;

//     // Null out fields not being used
//     (*iter)->_reduce_dims = NULL;
//     (*iter)->keepdims = NULL;
//     (*iter)->reduce_shape = NULL;
//     (*iter)->reduce_strides = NULL;

//     return true;


// }

// static bool Iter_create_reduction_iterator(Iter **iter, ArrayList *list, IterCtx *ctx, IterFlags flag) {
//     //WRITE("1.I\n")
//     *iter = malloc(sizeof(Iter));
//     if (*iter == NULL) { WRITE("Iter == NULL after malloc\n")return false; }

//     uint64_t *iter_shape;
//     uint64_t iter_ndim;

//     int64_t **iter_strides;
//     int64_t **iter_backstrides;
//     int64_t *iter_inner_strides;

//     uint64_t iter_inner_loop_size = 1;
//     uint64_t iter_total_size = 1;

//     uint64_t *iter_coordinates;

//     uint64_t outer_ndim;
//     uint64_t inner_ndim;
    
//     int64_t nops = list->noe;


//     (*iter)->baseptrs = malloc(sizeof(char*) * nops);
//     (*iter)->dataptrs = malloc(sizeof(char*) * nops);
//     if (!(*iter)->baseptrs || !(*iter)->dataptrs) {WRITE("Iter data and base ptrs == NULL after malloc\n")  return false; }

//     for (int i = 0; i < nops; i++) {
//         (*iter)->baseptrs[i] = list->arrays[i]->data;
//         (*iter)->dataptrs[i] = list->arrays[i]->data;
//     }

//     //WRITE("2.I\n")

//     uint64_t input_ndim = ctx->input_ndim;
//       //  printf("Input_ndim:%lld\n", input_ndim);
//     uint64_t *input_shape = ctx->input_shape;

//     uint64_t reduction_ndims = ctx->reduction_ndims;
//     uint64_t num_reduced = ctx->num_reduced;

//     uint64_t *reduction_perm = ctx->reduction_perm;

//     nops = list->noe;
//     iter_ndim = input_ndim;

//     iter_shape = malloc(sizeof(uint64_t) * input_ndim);
//     int64_t *iter_reduce_shape = malloc(sizeof(int64_t) * num_reduced);
//     int64_t **iter_reduce_strides = malloc(sizeof(int64_t *) * nops);
//     int64_t *iter_flat_strides = malloc(sizeof(int64_t) * iter_ndim);

//     iter_coordinates = calloc(sizeof(uint64_t), iter_ndim);
//     if (!iter_coordinates) { return false; }

//     for (uint64_t i = 0; i < input_ndim; i++) {
//         iter_shape[i] = input_shape[reduction_perm[i]];
//     }

//     //WRITE("2.IR\n")

//     if (arraylist_broadcast_stride_from_shape(list, iter_shape, iter_ndim, &iter_strides) == false) { return false; }

//     for (int ar = 0; ar < nops; ar++) {
//         int64_t *old = iter_strides[ar];
//         int64_t *new = malloc(sizeof(int64_t) * input_ndim);

//         for (uint64_t i = 0; i < input_ndim; i++) {
//             new[i] = old[reduction_perm[i]];
//         }

//         iter_strides[ar] = new;

//     }

//     //WRITE("3.IR\n")

//     iter_backstrides = malloc(sizeof(int64_t *) * nops);
//     if (!iter_backstrides) { WRITE("Failed to create iter_backstrides\n")return false; }

//     for (int ar = 0; ar < nops; ar++) {
//         iter_backstrides[ar] = malloc(sizeof(int64_t) * input_ndim);
//         if (!iter_backstrides[ar]) { return false; }

//         for (uint64_t d = 0; d < input_ndim; d++) {
//             iter_backstrides[ar][d] = iter_strides[ar][d] * ((int64_t)iter_shape[d] - 1);
//         }
//     }

//     //WRITE("4.IR\n")

//     for (int ar = 0; ar < nops; ar++) {
//         iter_reduce_strides[ar] = malloc(sizeof(int64_t) * num_reduced);
//     }

//     //WRITE("5.IR\n")

//     for (int r = 0; r < num_reduced; r++) {
//         iter_reduce_shape[r] = iter_shape[input_ndim - num_reduced + r];
//     }

//     //WRITE("6.IR\n")

//     for (int ar = 0; ar < nops; ar++) {
//         for (int r = 0; r < num_reduced; r++) {
//             iter_reduce_strides[ar][r] = iter_strides[ar][input_ndim - num_reduced + r];
//         }
//     }

//     //WRITE("7.IR\n")

//     int out_idx = nops - 1;
//     for (uint64_t i = input_ndim - num_reduced; i < input_ndim; i++) {
//         iter_strides[out_idx][i] = 0;
//     }

//    // WRITE("8.IR\n")

//     int64_t curr_stride = 1;
//     for (int i = num_reduced - 1; i > 0; i--) {
//         iter_flat_strides[i] = curr_stride;
//         curr_stride *= i * iter_reduce_shape[i];
//     }

//     //WRITE("9.IR\n")

//     iter_total_size = 1;
//     for(uint64_t i = 0; i < iter_ndim; i++) {
//         iter_total_size *= iter_shape[i];
//     }    

//     //WRITE("10.IR\n")

//     iter_inner_loop_size = 1;
//     for (uint64_t i = input_ndim - num_reduced; i < input_ndim; i++) {
//         iter_inner_loop_size *= iter_shape[i];
//        // printf("iter_shape[%lld]: %lld, This is inner_loop_size: %lld\n", i, iter_shape[i], iter_inner_loop_size);
//     }

//    // WRITE("11.IR\n")
//     iter_ndim = input_ndim - num_reduced;
//     (*iter)->rndims = num_reduced;
//     (*iter)->reduce_shape = iter_reduce_shape;
//     (*iter)->reduce_strides = iter_reduce_strides;
//     (*iter)->flat_strides = iter_flat_strides;

// //    WRITE("12.IR\n")
        
//     outer_ndim = input_ndim - num_reduced;
//     inner_ndim = num_reduced;

    
//    // WRITE("I.0\n")
//     (*iter)->nops = nops;
//     (*iter)->ndim = iter_ndim;
//     (*iter)->shape = iter_shape;
//     (*iter)->strides = iter_strides;
//     (*iter)->backstrides = iter_backstrides;
//     (*iter)->coordinates = iter_coordinates;
//     (*iter)->total_size = iter_total_size;
//     (*iter)->outer_ndim = outer_ndim;
//     (*iter)->inner_ndim = inner_ndim;
//     (*iter)->inner_loop_size = iter_inner_loop_size;
//     (*iter)->inner_strides = iter_inner_strides;
//     (*iter)->counter = 0;

//     return true;

// }

// static bool Iter_create_matmul_iterator(Iter **iter, ArrayList *list, IterCtx *ctx, IterFlags flag) {
//     if (!iter || !list || !ctx) {
//         REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "iter, list, or ctx is NULL");
//         error_stack_print();
//         return false;
//     }

//     *iter = calloc(1, sizeof(Iter));
//     if (!(*iter)) {
//         REPORT_ERROR(ERR_MALLOC_FAILED, "failed to calloc Iter");
//         return false;
//     }

//     int nops = list->noe;

//     uint64_t M = ctx->arr_a_row_mm;
//     uint64_t K = ctx->arr_a_col_mm;
//     uint64_t N = ctx->arr_b_col_mm;

//     uint64_t iter_ndim = list->arrays[2]->ndim;
//     uint64_t *iter_shape = malloc(iter_ndim * sizeof(uint64_t));

//     memcpy(iter_shape, list->arrays[2]->shape, sizeof(uint64_t) * iter_ndim);



//     // if (!arraylist_broadcast_for_matmul(list, &iter_shape, &iter_ndim)) {
//     //     REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_broadcast_for_matmul");
//     //     free(*iter);
//     //     *iter = NULL;
//     //     error_stack_print();
//     //     return false;
//     // }

//     uint64_t outer_ndim = (iter_ndim >= 2) ? iter_ndim - 2 : 0;

//     int64_t **batch_strides = malloc(sizeof(int64_t*) * nops);
//     int64_t **batch_backstrides = malloc(sizeof(int64_t*) * nops);

//     if (!batch_strides || !batch_backstrides) {
//         REPORT_ERROR(ERR_MALLOC_FAILED, "failed to malloc batch_strides or batch_backstrides");
//         free(batch_strides);
//         free(batch_backstrides);
//         free(iter_shape);
//         free(*iter);
//         *iter = NULL;
//         error_stack_print();
//         return false;
//     }

//     // zero them so partial-init is safe to free on error
//     memset(batch_strides,     0, sizeof(int64_t*) * nops);
//     memset(batch_backstrides, 0, sizeof(int64_t*) * nops);

//     for (int ar = 0; ar < nops; ar++) {
//         batch_strides[ar]     = calloc(outer_ndim, sizeof(int64_t));
//         batch_backstrides[ar] = calloc(outer_ndim, sizeof(int64_t));

//         if (!batch_strides[ar] || !batch_backstrides[ar]) {
//             REPORT_ERROR(ERR_MALLOC_FAILED, "failed to calloc batch_strides[%d] or batch_backstrides[%d]", ar, ar);
//             error_stack_print();
//             goto fail;
//         }

//         array *arr = list->arrays[ar];
//         if (!arr) {
//             REPORT_ERROR(ERR_STRUCT_NOT_INIT, "list->arrays[%d] is NULL", ar);
//             error_stack_print();
//             goto fail;
//         }

//         int64_t arr_batch_ndim = (arr->ndim >= 2) ? arr->ndim - 2 : 0;
//         int64_t offset = (int64_t)outer_ndim - arr_batch_ndim;

//         for (int64_t dim = 0; dim < (int64_t)outer_ndim; dim++) {
//             int64_t arr_dim = dim - offset;
//             uint64_t arr_shape_here = (arr_dim >= 0) ? arr->shape[arr_dim] : 1;

//             if (arr_shape_here == 1 && iter_shape[dim] > 1) {
//                 batch_strides[ar][dim]     = 0;
//                 batch_backstrides[ar][dim] = 0;
//             } else if (arr_dim >= 0) {
//                 batch_strides[ar][dim]     = arr->stride[arr_dim];
//                 batch_backstrides[ar][dim] = arr->stride[arr_dim] * ((int64_t)iter_shape[dim] - 1);
//             } else {
//                 batch_strides[ar][dim]     = 0;
//                 batch_backstrides[ar][dim] = 0;
//             }
//         }
//     }

//     uint64_t total_batches = 1;
//     for (uint64_t d = 0; d < outer_ndim; d++)
//         total_batches *= iter_shape[d];

//     (*iter)->baseptrs = malloc(sizeof(char*) * nops);
//     (*iter)->dataptrs = malloc(sizeof(char*) * nops);

//     if (!(*iter)->baseptrs || !(*iter)->dataptrs) {
//         REPORT_ERROR(ERR_MALLOC_FAILED, "failed to malloc baseptrs or dataptrs");
//         error_stack_print();
//         goto fail;
//     }

//     for (int i = 0; i < nops; i++) {
//         if (!list->arrays[i] || !list->arrays[i]->data) {
//             REPORT_ERROR(ERR_STRUCT_FIELD_NOT_INIT, "list->arrays[%d]->data is NULL", i);
//             error_stack_print();
//             goto fail;
//         }
//         (*iter)->baseptrs[i] = list->arrays[i]->data;
//         (*iter)->dataptrs[i] = list->arrays[i]->data;
//     }

//     (*iter)->coordinates = calloc(outer_ndim, sizeof(uint64_t));
//     if (!(*iter)->coordinates) {
//         REPORT_ERROR(ERR_MALLOC_FAILED, "failed to calloc coordinates");
//         error_stack_print();
//         goto fail;
//     }

//     (*iter)->nops            = nops;
//     (*iter)->ndim            = iter_ndim;
//     (*iter)->shape           = iter_shape;
//     (*iter)->strides         = batch_strides;
//     (*iter)->backstrides     = batch_backstrides;
//     (*iter)->outer_ndim      = outer_ndim;
//     (*iter)->inner_ndim      = 2;
//     (*iter)->inner_loop_size = 1;
//     (*iter)->total_size      = total_batches;
//     (*iter)->counter         = 0;
//     (*iter)->arr_a_row_mm    = M;
//     (*iter)->arr_a_col_mm    = K;
//     (*iter)->arr_b_col_mm    = N;
//     (*iter)->_reduce_dims    = NULL;
//     (*iter)->keepdims        = false;
//     (*iter)->reduce_shape    = NULL;
//     (*iter)->reduce_strides  = NULL;

//     Iter_print((*iter));
//     return true;

// fail:
//     for (int ar = 0; ar < nops; ar++) {
//         if (batch_strides[ar])     free(batch_strides[ar]);
//         if (batch_backstrides[ar]) free(batch_backstrides[ar]);
//     }
//     free(batch_strides);
//     free(batch_backstrides);
//     free(iter_shape);
//     if ((*iter)->baseptrs) free((*iter)->baseptrs);
//     if ((*iter)->dataptrs) free((*iter)->dataptrs);
//     if ((*iter)->coordinates) free((*iter)->coordinates);
//     free(*iter);
//     *iter = NULL;
//     return false;
// }

// bool Iter_create_iterator(Iter **iter, ArrayList *list, IterCtx *ctx, IterFlags flag) {
//     if (!list) {printf("List is false\n"); return false; }

//     switch (flag) {
//         case ITER_ELEMENTWISE:
//             if (Iter_create_elementwise_iterator(iter, list, ctx, flag) == false) {
//                 REPORT_ERROR(ERR_FUNC_CALL_FAILED, "Iter_create_elementwise_iterator");
//                 return false;
//             }
//             return true;
//         break;

//         case ITER_REDUCTION:
//             if (Iter_create_reduction_iterator(iter, list, ctx, flag) == false) {
//                 REPORT_ERROR(ERR_FUNC_CALL_FAILED, "Iter_create_reduction_iterator");
//                 return false;
//             }
//             return true;


//         case ITER_BATCHED_MATMUL:
//             if (Iter_create_matmul_iterator(iter, list, ctx, flag) == false) {
//                 REPORT_ERROR(ERR_FUNC_CALL_FAILED, "Iter_create_matmul_iterator");
//                 return false;
//             }
//             return true;
//         break;
//     }
// }

// bool Iter_next_ele(Iter *iter) {
//  //   WRITE("1\n")
//     if (!iter) return false;

//     if (iter->counter + 1 >= iter->total_size) {
//         return false;
//     }
    
//     iter->counter++;
//  //   WRITE("2\n")

// //    printf("Iter ndims:%lld\n", iter->ndim);
// //    printf("Iter outer_ndims:%llu\n", iter->outer_ndim);



//     for (int64_t dim = iter->ndim - 1; dim >= 0; dim--) {

//         if (iter->coordinates[dim] + 1 < iter->shape[dim]) {
//             iter->coordinates[dim]++;
            
//             for (int arr = 0; arr < iter->nops; arr++) {
//                 iter->dataptrs[arr] += iter->strides[arr][dim];
//             }

//             return true;

//         }
        
//         iter->coordinates[dim] = 0;
//         for (int arr = 0; arr < iter->nops; arr++) {
//             iter->dataptrs[arr] -= iter->backstrides[arr][dim];
//         }
        
//     }
//  //   WRITE("3\n")

//     return false;

// }

// bool Iter_next_chunk(Iter *iter) {
//    if (!iter) { printf("Iterator is false\n"); return false; }

//    iter->counter += iter->inner_loop_size;

//     if (iter->counter >= iter->total_size) {
//         //WRITE("DEBUG: iter->counter >= iter->total_size\n")
//         return false;
//     }

//     for (int64_t dim = iter->outer_ndim - 1; dim >= 0; dim--) {
//         if (iter->coordinates[dim] + 1 < iter->shape[dim]) {
            
//             iter->coordinates[dim]++;
 
//             for (int arr = 0; arr < iter->nops; arr++) {
//                 iter->dataptrs[arr] += iter->strides[arr][dim];
//             }
//             return true;

//         }
        
//         iter->coordinates[dim] = 0;
//         for (int arr = 0; arr < iter->nops; arr++) {
//             iter->dataptrs[arr] -= iter->backstrides[arr][dim];

//         }
        
//     }
    
//     return false;

// }

// bool Iter_run_kernel(Iter *iter, kernel_fn func) {
//     if (!iter) { printf("iter is null\n");return false; }

//     kernel_info info = {
//         .dataptrs = iter->dataptrs,
//         .inner_loop_size = iter->inner_loop_size,
//         .inner_strides = iter->inner_strides,
//         .rndims = iter->rndims,
//         .reduce_shape = iter->reduce_shape,
//         .reduce_strides = iter->reduce_strides,
//         .flat_strides = iter->flat_strides,
//         .arr_a_col_mm = iter->arr_a_col_mm,
//         .arr_a_row_mm = iter->arr_a_row_mm,
//         .arr_b_col_mm = iter->arr_b_col_mm,
//         .arr_b_row_mm = iter->arr_b_row_mm
//     };

//     if (iter->total_size == 0) {
//         return true; 
//     }

//     do {
        
//         func(&info);

//     } while (Iter_next_chunk(iter));

//     return true;

// }

// bool Iter_has_next(Iter *iter) {
//     return iter->counter < iter->total_size;
// }

// bool Iter_get_ndim(Iter *iter, uint64_t *out_ndim) {
//     if (!iter || !out_ndim) {
//         return false;
//     }
//     *out_ndim = iter->ndim;
//     return true;
// }

// bool Iter_get_shape(Iter *iter, uint64_t **out_shape) {
//     if (!iter || !out_shape || !iter->shape) {
//         return false;
//     }

//     *out_shape = malloc(sizeof(uint64_t) * iter->ndim);
//     if (!*out_shape) {
//         return false;
//     }

//     memcpy(*out_shape, iter->shape, sizeof(uint64_t) * iter->ndim);
//     return true;
// }

// bool Iter_get_stride(Iter *iter, int op_index, int64_t **out_stride) {
//     if (!iter  || op_index < 0 || op_index >= iter->nops) {
//         return false;
//     }
    
//     if (!iter->strides || !iter->strides[op_index]) {
//         return false;
//     }

//     *out_stride = malloc(sizeof(int64_t) * iter->ndim);
//     if (!*out_stride) {
//         return false;
//     }

//     memcpy(*out_stride, iter->strides[op_index], sizeof(int64_t) * iter->ndim);
//     return true;
// }

// bool Iter_free(Iter *iter) {
//     if (!iter) return false;

//     free(iter->shape);
//     free(iter->baseptrs);
//     free(iter->dataptrs);
//     free(iter->coordinates);
//     free(iter->inner_strides);
//     free(iter->_reduce_dims);

//     if (iter->strides) {
//         for (int i = 0; i < iter->nops; i++) {
//             free(iter->strides[i]);
//         }
//         free(iter->strides);
//     }

//     if (iter->backstrides) {
//         for (int i = 0; i < iter->nops; i++) {
//             free(iter->backstrides[i]);
//         }
//         free(iter->backstrides);
//     }

//     free(iter);

//     return true;
// }