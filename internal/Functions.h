#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "array.h"
#include "Ops.h"
#include "Iter.h"
#include "Engine.h"

// Creation
bool array_zeros(array **out_array, uint64_t ndim, uint64_t *shape, dtype type, Backend backend);
bool array_zeros_like(array *in, array **out_array);
bool array_ones(array **out_array, uint64_t ndim, uint64_t *shape, dtype type, Backend backend);
bool array_ones_like(array *in, array **out_array);
bool array_fill(array **out_array, uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double fill_value);
bool array_arange(array **out_array, uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double start, double step);
bool array_linspace(array **out_array, uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double start, double stop);
bool array_randu(array **out_array, uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double low, double high);
bool array_randn(array **out_array, uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double mean, double stdev);
bool array_randint(array **out_array, uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double low, double high);

// Arithmetic
bool array_add(array **out_array, array *a, array *b);
bool array_subtract(array **out_array, array *a, array *b);
bool array_multiply(array **out_array, array *a, array *b);
bool array_divide(array **out_array, array *a, array *b);
bool array_floor_divide(array **out_array, array *a, array *b);
bool array_power(array **out_array, array *a, array *b);

// Logical
bool array_equality(array **out_array, array *a, array *b);
bool array_greater_than(array **out_array, array *a, array *b);
bool array_less_than(array **out_array, array *a, array *b);
bool array_maximum(array **out_array, array *a, array *b);
bool array_minimum(array **out_array, array *a, array *b);

// Unary
bool array_square_root(array **out_array, array *a);
bool array_absolute_value(array **out_array, array *a);
bool array_negation(array **out_array, array *a);


// Reductions

bool array_sum(array **out_array, array *a, uint64_t rndim, uint64_t *dims, bool keepdim);
bool array_max(array **out_array, array *a, uint64_t rndim, uint64_t *dims, bool keepdim);
bool array_min(array **out_array, array *a, uint64_t rndim, uint64_t *dims, bool keepdim);

// Matmul

bool array_matmul(array **out_array, array *a, array *b);


// Utility
bool array_print_array(array *array);


// Arithmetic Backward
bool array_add_backward(Function *fn);
bool array_subtract_backward(Function *fn);
bool array_multiply_backward(Function *fn);
bool array_divide_backward(Function *fn);
bool array_power_backward(Function *fn);



#endif // FUNCTIONS_H