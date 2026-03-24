#ifndef ITER_H
#define ITER_H

#include <inttypes.h>
#include <stdbool.h>
#include "arraylist.h"

typedef enum {
    ITER_ELEMENTWISE,
    ITER_REDUCTION,
    ITER_BATCHED_MATMUL
} IterFlags;

// typedef enum {
//     ITER_ELEMENTWISE,
//     ITER_REDUCTION,
//     ITER_BATCHED_MATMUL
// } IterType;
typedef struct kernel_info {
    char **dataptrs;
    char **saved_dataptrs;
    int64_t *inner_strides;
    int64_t *saved_dataptrs_inner_strides;
    uint64_t inner_loop_size;
    uint64_t rndims;              // Reductions ndims
    int64_t *reduce_shape;          
    int64_t **reduce_strides;
    int64_t *flat_strides;
    uint64_t arr_a_row_mm;
    uint64_t arr_a_col_mm;
    uint64_t arr_b_row_mm;
    uint64_t arr_b_col_mm;
} kernel_info;

typedef void (*kernel_fn) (kernel_info *info);

typedef struct Iter Iter;


typedef struct Iter {
    IterFlags type;
    bool (*next)(Iter *);
    bool (*next_ele)(Iter *);
    bool (*run)(Iter *, kernel_fn);
} Iter;


typedef struct ElementwiseIter{
    Iter base;
    int nops;                   // Number of operands
    char **baseptrs;            // Ptrs to operands data  
    char **dataptrs;           // Ptrs to operands data during iteration
    int64_t **strides;          // Strides for each operand [nops][ndim]
    int64_t **backstrides;      // For resetting dimensions [nops][ndim]
    int64_t *inner_strides;       // Strides for loop in kernel [nops]
    uint64_t ndim;              // Number of dimensions
    uint64_t *shape;            // Iterator Shape
    uint64_t *coordinates;     // Tracks position in iteration space
    uint64_t counter;          // Tracks linear index
    uint64_t total_size;       // Total number of elements to iterate through, product of iterator shape
    uint64_t inner_loop_size;     // Size of innermost contiguous loop
    uint64_t outer_ndim;
    uint64_t inner_ndim;
    bool *_reduce_dims;           // Dimensions to reduce over
    bool keepdims;                // Wether to keep reduction dimensions
    uint64_t rndims;              // Reductions ndims
    int64_t *reduce_shape;          
    int64_t **reduce_strides;
    int64_t *flat_strides;

}ElementwiseIter;


typedef struct ReductionIter{
    Iter base;
    int nops;                   // Number of operands
    char **baseptrs;            // Ptrs to operands data  
    char **dataptrs;            // Ptrs to operands data during iteration
    int64_t *reduce_shape;     
    int64_t **strides;          // Strides for each operand [nops][ndim]
    int64_t **backstrides;      // For resetting dimensions [nops][ndim]
    int64_t **reduce_strides;
    int64_t *flat_strides;
    uint64_t ndim;              // Input dimensions == Iter dimensions
    uint64_t *shape;            // Iterator Shape
    uint64_t *coordinates;     // Tracks position in iteration space
    uint64_t counter;          // Tracks linear index
    uint64_t total_size;       // Total number of elements to iterate through, product of iterator shape
    uint64_t inner_loop_size;     // Size of innermost contiguous loop
    uint64_t outer_ndim;
    uint64_t inner_ndim;
    uint64_t *input_shape;
    uint64_t reduction_ndims;
    uint64_t *reduction_perm;
    uint64_t num_reduced;
}ReductionIter;


typedef struct MatmulIter{
    Iter base;
    int nops;
    char **baseptrs;
    char **dataptrs;
    int64_t **strides;
    int64_t **backstrides;
    uint64_t ndim;
    uint64_t counter;
    uint64_t total_size;
    uint64_t inner_loop_size;
    uint64_t outer_ndim;
    uint64_t inner_ndim;
    uint64_t arr_a_row_mm;
    uint64_t arr_a_col_mm;
    uint64_t arr_b_col_mm;
    uint64_t *shape;
    uint64_t *coordinates;
}MatmulIter;


typedef struct IterCtx{
    uint64_t input_ndim;
    uint64_t *input_shape;
    uint64_t reduction_ndims;
    uint64_t *reduction_perm;
    uint64_t num_reduced;
    uint64_t len_of_inner_ndim;
    uint64_t arr_a_row_mm;
    uint64_t arr_a_col_mm;
    uint64_t arr_b_row_mm;
    uint64_t arr_b_col_mm;
}IterCtx;



bool Iter_create_iterator(Iter **iter, ArrayList *list, IterCtx *ctx, IterFlags flag);


#endif // ITER_H