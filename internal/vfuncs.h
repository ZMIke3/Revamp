#ifndef VFUNCTIONS_H
#define VFUNCTIONS_H

#include "Engine.h"
#include "array.h"
#include "Functions.h"


bool value_zeros(Value **out_value, uint64_t ndim, uint64_t *shape, dtype type, Backend backend, bool requires_grad);
bool value_zeros_like(Value **out_value, Value *in, bool requires_grad);
bool value_ones(Value **out_value, uint64_t ndim, uint64_t *shape, dtype type, Backend backend, bool requires_grad);
bool value_ones_like(Value **out_value, Value *in, bool requires_grad);

bool value_fill(Value **out_value, uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double fill_value, bool requires_grad);
bool value_arange(Value **out_value, uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double start, double step, bool requires_grad);
bool value_linspace(Value **out_value, uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double start, double stop, bool requires_grad);
bool value_randu(Value **out_value, uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double low, double high, bool requires_grad);
bool value_randn(Value **out_value, uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double mean, double stdev, bool requires_grad);
bool value_randint(Value **out_value, uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double low, double high, bool requires_grad);

bool value_add(Value **out_value, Value *a, Value *b);
bool value_subtract(Value **out_value, Value *a, Value *b);
bool value_multiply(Value **out_value, Value *a, Value *b);
bool value_divide(Value **out_value, Value *a, Value *b);
bool value_floor_divide(Value **out_value, Value *a, Value *b);
bool value_power(Value **out_value, Value *a, Value *b);

bool value_equality(Value **out_value, Value *a, Value *b);
bool value_greater_than(Value **out_value, Value *a, Value *b);
bool value_less_than(Value **out_value, Value *a, Value *b);
bool value_maximum(Value **out_value, Value *a, Value *b);
bool value_minimum(Value **out_value, Value *a, Value *b);

bool value_square_root(Value **out_value, Value *a);
bool value_absolute_value(Value **out_value, Value *a);
bool value_negation(Value **out_value, Value *a);

bool value_sum(Value **out_value, Value *a, uint64_t rndim, uint64_t *dims, bool keepdim);
bool value_max(Value **out_value, Value *a, uint64_t rndim, uint64_t *dims, bool keepdim);
bool value_min(Value **out_value, Value *a, uint64_t rndim, uint64_t *dims, bool keepdim);

bool value_matmul(Value **out_value, Value *a, Value *b);

#endif // VFUNCTIONS_H

