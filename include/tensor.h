#ifndef TENSOR_H
#define TENSOR_H

#include "Engine.h"
#include "vfuncs.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct Tensor {
    Value *self;
} Tensor;

Tensor *tensor_create(uint64_t ndim, uint64_t *shape, Backend backend, dtype type, void *data);


Tensor *tensor_zeros(uint64_t ndim, uint64_t *shape, dtype type, Backend backend, bool requires_grad);
Tensor *tensor_zeros_like(Tensor *a, bool requires_grad);
Tensor *tensor_ones(uint64_t ndim, uint64_t *shape, dtype type, Backend backend, bool requires_grad);
Tensor *tensor_ones_like(Tensor *a, bool requires_grad);
Tensor *tensor_fill(uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double fill_value, bool requires_grad);
Tensor *tensor_arange(uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double start, double step, bool requires_grad);
Tensor *tensor_linspace(uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double start, double stop, bool requires_grad);
Tensor *tensor_randu(uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double low, double high, bool requires_grad);
Tensor *tensor_randn(uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double mean, double stdev, bool requires_grad);
Tensor *tensor_randint(uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double low, double high, bool requires_grad);

Tensor *tensor_add(Tensor *a, Tensor *b);
Tensor *tensor_sub(Tensor *a, Tensor *b);
Tensor *tensor_mul(Tensor *a, Tensor *b);
Tensor *tensor_div(Tensor *a, Tensor *b);
Tensor *tensor_floor_div(Tensor *a, Tensor *b);
Tensor *tensor_pow(Tensor *a, Tensor *b);

Tensor *tensor_eq(Tensor *a, Tensor *b);
Tensor *tensor_gt(Tensor *a, Tensor *b);
Tensor *tensor_lt(Tensor *a, Tensor *b);
Tensor *tensor_maximum(Tensor *a, Tensor *b);
Tensor *tensor_minimum(Tensor *a, Tensor *b);

Tensor *tensor_sqrt(Tensor *a);
Tensor *tensor_abs(Tensor *a);
Tensor *tensor_neg(Tensor *a);

Tensor *tensor_sum(Tensor *a, uint64_t rndim, uint64_t *dims, bool keepdim);
Tensor *tensor_max(Tensor *a, uint64_t rndim, uint64_t *dims, bool keepdim);
Tensor *tensor_min(Tensor *a, uint64_t rndim, uint64_t *dims, bool keepdim);

Tensor *tensor_matmul(Tensor *a, Tensor *b);

bool grad(Tensor *a);
bool tensor_retain_grad(Tensor *a);
void tensor_print(Tensor *a);
void tensor_print_grad(Tensor *a);

#endif // TENSOR_H