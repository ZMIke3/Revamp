#ifndef BACKWARDS_KERNELS_H
#define BACKWARDS_KERNELS_H

#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "Iter.h"

void f32_kernel_add_backward(kernel_info *info);
void f64_kernel_add_backward(kernel_info *info);

void f32_kernel_subtract_backward(kernel_info *info);
void f64_kernel_subtract_backward(kernel_info *info);

void f32_kernel_multiply_backward(kernel_info *info);
void f64_kernel_multiply_backward(kernel_info *info);

void f32_kernel_divide_backward(kernel_info *info);
void f64_kernel_divide_backward(kernel_info *info);

void f64_kernel_power_backward(kernel_info *info);


#endif // BACKWARDS_KERNELS_H
