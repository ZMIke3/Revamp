#ifndef KERNELS_H
#define KERNELS_H

#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <cblas.h>
#include "Iter.h"

typedef struct kernel_info kernel_info;

void i8_kernel_add(kernel_info *info);
void i16_kernel_add(kernel_info *info);
void i32_kernel_add(kernel_info *info);
void i64_kernel_add(kernel_info *info);
void u8_kernel_add(kernel_info *info);
void u16_kernel_add(kernel_info *info);
void u32_kernel_add(kernel_info *info);
void u64_kernel_add(kernel_info *info);
void f32_kernel_add(kernel_info *info);
void f64_kernel_add(kernel_info *info);

void i8_kernel_subtract(kernel_info *info);
void i16_kernel_subtract(kernel_info *info);
void i32_kernel_subtract(kernel_info *info);
void i64_kernel_subtract(kernel_info *info);
void u8_kernel_subtract(kernel_info *info);
void u16_kernel_subtract(kernel_info *info);
void u32_kernel_subtract(kernel_info *info);
void u64_kernel_subtract(kernel_info *info);
void f32_kernel_subtract(kernel_info *info);
void f64_kernel_subtract(kernel_info *info);

void i8_kernel_multiply(kernel_info *info);
void i16_kernel_multiply(kernel_info *info);
void i32_kernel_multiply(kernel_info *info);
void i64_kernel_multiply(kernel_info *info);
void u8_kernel_multiply(kernel_info *info);
void u16_kernel_multiply(kernel_info *info);
void u32_kernel_multiply(kernel_info *info);
void u64_kernel_multiply(kernel_info *info);
void f32_kernel_multiply(kernel_info *info);
void f64_kernel_multiply(kernel_info *info);

void i8_kernel_divide(kernel_info *info);
void i16_kernel_divide(kernel_info *info);
void i32_kernel_divide(kernel_info *info);
void i64_kernel_divide(kernel_info *info);
void u8_kernel_divide(kernel_info *info);
void u16_kernel_divide(kernel_info *info);
void u32_kernel_divide(kernel_info *info);
void u64_kernel_divide(kernel_info *info);
void f32_kernel_divide(kernel_info *info);
void f64_kernel_divide(kernel_info *info);

void i8_kernel_equals(kernel_info *info);
void i16_kernel_equals(kernel_info *info);
void i32_kernel_equals(kernel_info *info);
void i64_kernel_equals(kernel_info *info);
void u8_kernel_equals(kernel_info *info);
void u16_kernel_equals(kernel_info *info);
void u32_kernel_equals(kernel_info *info);
void u64_kernel_equals(kernel_info *info);
void f32_kernel_equals(kernel_info *info);
void f64_kernel_equals(kernel_info *info);

void f32_kernel_negate(kernel_info *info);
void f64_kernel_negate(kernel_info *info);
void i16_kernel_negate(kernel_info *info);
void i32_kernel_negate(kernel_info *info);
void i64_kernel_negate(kernel_info *info);
void i8_kernel_negate(kernel_info *info);

void f32_kernel_abs(kernel_info *info);
void f64_kernel_abs(kernel_info *info);
void i16_kernel_abs(kernel_info *info);
void i32_kernel_abs(kernel_info *info);
void i64_kernel_abs(kernel_info *info);
void i8_kernel_abs(kernel_info *info);

void i8_kernel_sum(kernel_info *info);
void i16_kernel_sum(kernel_info *info);
void i32_kernel_sum(kernel_info *info);
void i64_kernel_sum(kernel_info *info);
void u8_kernel_sum(kernel_info *info);
void u16_kernel_sum(kernel_info *info);
void u32_kernel_sum(kernel_info *info);
void u64_kernel_sum(kernel_info *info);
void f32_kernel_sum(kernel_info *info);
void f64_kernel_sum(kernel_info *info);

void i8_kernel_max(kernel_info *info);
void i16_kernel_max(kernel_info *info);
void i32_kernel_max(kernel_info *info);
void i64_kernel_max(kernel_info *info);
void u8_kernel_max(kernel_info *info);
void u16_kernel_max(kernel_info *info);
void u32_kernel_max(kernel_info *info);
void u64_kernel_max(kernel_info *info);
void f32_kernel_max(kernel_info *info);
void f64_kernel_max(kernel_info *info);

void i8_kernel_min(kernel_info *info);
void i16_kernel_min(kernel_info *info);
void i32_kernel_min(kernel_info *info);
void i64_kernel_min(kernel_info *info);
void u8_kernel_min(kernel_info *info);
void u16_kernel_min(kernel_info *info);
void u32_kernel_min(kernel_info *info);
void u64_kernel_min(kernel_info *info);
void f32_kernel_min(kernel_info *info);
void f64_kernel_min(kernel_info *info);

void f32_kernel_matmul(kernel_info *info);
void f64_kernel_matmul(kernel_info *info);

void i8_kernel_argmax(kernel_info *info);
void i16_kernel_argmax(kernel_info *info);
void i32_kernel_argmax(kernel_info *info);
void i64_kernel_argmax(kernel_info *info);
void u8_kernel_argmax(kernel_info *info);
void u16_kernel_argmax(kernel_info *info);
void u32_kernel_argmax(kernel_info *info);
void u64_kernel_argmax(kernel_info *info);
void f32_kernel_argmax(kernel_info *info);
void f64_kernel_argmax(kernel_info *info);

void i8_to_i16_cast(kernel_info *info);
void i8_to_i32_cast(kernel_info *info);
void i8_to_i64_cast(kernel_info *info);
void i8_to_u8_cast(kernel_info *info);
void i8_to_u16_cast(kernel_info *info);
void i8_to_u32_cast(kernel_info *info);
void i8_to_u64_cast(kernel_info *info);
void i8_to_f32_cast(kernel_info *info);
void i8_to_f64_cast(kernel_info *info);
void i8_to_bool_cast(kernel_info *info);

void i16_to_i8_cast(kernel_info *info);
void i16_to_i32_cast(kernel_info *info);
void i16_to_i64_cast(kernel_info *info);
void i16_to_u8_cast(kernel_info *info);
void i16_to_u16_cast(kernel_info *info);
void i16_to_u32_cast(kernel_info *info);
void i16_to_u64_cast(kernel_info *info);
void i16_to_f32_cast(kernel_info *info);
void i16_to_f64_cast(kernel_info *info);
void i16_to_bool_cast(kernel_info *info);

void i32_to_i8_cast(kernel_info *info);
void i32_to_i16_cast(kernel_info *info);
void i32_to_i64_cast(kernel_info *info);
void i32_to_u8_cast(kernel_info *info);
void i32_to_u16_cast(kernel_info *info);
void i32_to_u32_cast(kernel_info *info);
void i32_to_u64_cast(kernel_info *info);
void i32_to_f32_cast(kernel_info *info);
void i32_to_f64_cast(kernel_info *info);
void i32_to_bool_cast(kernel_info *info);

void i64_to_i8_cast(kernel_info *info);
void i64_to_i16_cast(kernel_info *info);
void i64_to_i32_cast(kernel_info *info);
void i64_to_u8_cast(kernel_info *info);
void i64_to_u16_cast(kernel_info *info);
void i64_to_u32_cast(kernel_info *info);
void i64_to_u64_cast(kernel_info *info);
void i64_to_f32_cast(kernel_info *info);
void i64_to_f64_cast(kernel_info *info);
void i64_to_bool_cast(kernel_info *info);

void u8_to_i8_cast(kernel_info *info);
void u8_to_i16_cast(kernel_info *info);
void u8_to_i32_cast(kernel_info *info);
void u8_to_i64_cast(kernel_info *info);
void u8_to_u16_cast(kernel_info *info);
void u8_to_u32_cast(kernel_info *info);
void u8_to_u64_cast(kernel_info *info);
void u8_to_f32_cast(kernel_info *info);
void u8_to_f64_cast(kernel_info *info);
void u8_to_bool_cast(kernel_info *info);

void u16_to_i8_cast(kernel_info *info);
void u16_to_i16_cast(kernel_info *info);
void u16_to_i32_cast(kernel_info *info);
void u16_to_i64_cast(kernel_info *info);
void u16_to_u8_cast(kernel_info *info);
void u16_to_u32_cast(kernel_info *info);
void u16_to_u64_cast(kernel_info *info);
void u16_to_f32_cast(kernel_info *info);
void u16_to_f64_cast(kernel_info *info);
void u16_to_bool_cast(kernel_info *info);

void u32_to_i8_cast(kernel_info *info);
void u32_to_i16_cast(kernel_info *info);
void u32_to_i32_cast(kernel_info *info);
void u32_to_i64_cast(kernel_info *info);
void u32_to_u8_cast(kernel_info *info);
void u32_to_u16_cast(kernel_info *info);
void u32_to_u64_cast(kernel_info *info);
void u32_to_f32_cast(kernel_info *info);
void u32_to_f64_cast(kernel_info *info);
void u32_to_bool_cast(kernel_info *info);

void u64_to_i8_cast(kernel_info *info);
void u64_to_i16_cast(kernel_info *info);
void u64_to_i32_cast(kernel_info *info);
void u64_to_i64_cast(kernel_info *info);
void u64_to_u8_cast(kernel_info *info);
void u64_to_u16_cast(kernel_info *info);
void u64_to_u32_cast(kernel_info *info);
void u64_to_f32_cast(kernel_info *info);
void u64_to_f64_cast(kernel_info *info);
void u64_to_bool_cast(kernel_info *info);

void f32_to_i8_cast(kernel_info *info);
void f32_to_i16_cast(kernel_info *info);
void f32_to_i32_cast(kernel_info *info);
void f32_to_i64_cast(kernel_info *info);
void f32_to_u8_cast(kernel_info *info);
void f32_to_u16_cast(kernel_info *info);
void f32_to_u32_cast(kernel_info *info);
void f32_to_u64_cast(kernel_info *info);
void f32_to_f64_cast(kernel_info *info);
void f32_to_bool_cast(kernel_info *info);

void f64_to_i8_cast(kernel_info *info);
void f64_to_i16_cast(kernel_info *info);
void f64_to_i32_cast(kernel_info *info);
void f64_to_i64_cast(kernel_info *info);
void f64_to_u8_cast(kernel_info *info);
void f64_to_u16_cast(kernel_info *info);
void f64_to_u32_cast(kernel_info *info);
void f64_to_u64_cast(kernel_info *info);
void f64_to_f32_cast(kernel_info *info);
void f64_to_bool_cast(kernel_info *info);

void bool_to_i8_cast(kernel_info *info);
void bool_to_i16_cast(kernel_info *info);
void bool_to_i32_cast(kernel_info *info);
void bool_to_i64_cast(kernel_info *info);
void bool_to_u8_cast(kernel_info *info);
void bool_to_u16_cast(kernel_info *info);
void bool_to_u32_cast(kernel_info *info);
void bool_to_u64_cast(kernel_info *info);
void bool_to_f32_cast(kernel_info *info);
void bool_to_f64_cast(kernel_info *info);



#endif // KERNELS_H
