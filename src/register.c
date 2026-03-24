#include "register.h"
void register_all_kernels(void) {

    dispatch_register_kernel(DTYPE_I8, CPU, KERNEL_ADD, i8_kernel_add);
    dispatch_register_kernel(DTYPE_I16, CPU, KERNEL_ADD, i16_kernel_add);
    dispatch_register_kernel(DTYPE_I32, CPU, KERNEL_ADD, i32_kernel_add);
    dispatch_register_kernel(DTYPE_I64, CPU, KERNEL_ADD, i64_kernel_add);
    dispatch_register_kernel(DTYPE_U8, CPU, KERNEL_ADD, u8_kernel_add);
    dispatch_register_kernel(DTYPE_U16, CPU, KERNEL_ADD, u16_kernel_add);
    dispatch_register_kernel(DTYPE_U32, CPU, KERNEL_ADD, u32_kernel_add);
    dispatch_register_kernel(DTYPE_U64, CPU, KERNEL_ADD, u64_kernel_add);
    dispatch_register_kernel(DTYPE_F32, CPU, KERNEL_ADD, f32_kernel_add);
    dispatch_register_kernel(DTYPE_F64, CPU, KERNEL_ADD, f64_kernel_add);

    dispatch_register_kernel(DTYPE_I8, CPU, KERNEL_SUBTRACT, i8_kernel_subtract);
    dispatch_register_kernel(DTYPE_I16, CPU, KERNEL_SUBTRACT, i16_kernel_subtract);
    dispatch_register_kernel(DTYPE_I32, CPU, KERNEL_SUBTRACT, i32_kernel_subtract);
    dispatch_register_kernel(DTYPE_I64, CPU, KERNEL_SUBTRACT, i64_kernel_subtract);
    dispatch_register_kernel(DTYPE_U8, CPU, KERNEL_SUBTRACT, u8_kernel_subtract);
    dispatch_register_kernel(DTYPE_U16, CPU, KERNEL_SUBTRACT, u16_kernel_subtract);
    dispatch_register_kernel(DTYPE_U32, CPU, KERNEL_SUBTRACT, u32_kernel_subtract);
    dispatch_register_kernel(DTYPE_U64, CPU, KERNEL_SUBTRACT, u64_kernel_subtract);
    dispatch_register_kernel(DTYPE_F32, CPU, KERNEL_SUBTRACT, f32_kernel_subtract);
    dispatch_register_kernel(DTYPE_F64, CPU, KERNEL_SUBTRACT, f64_kernel_subtract);

    dispatch_register_kernel(DTYPE_I8, CPU, KERNEL_MULTIPLY, i8_kernel_multiply);
    dispatch_register_kernel(DTYPE_I16, CPU, KERNEL_MULTIPLY, i16_kernel_multiply);
    dispatch_register_kernel(DTYPE_I32, CPU, KERNEL_MULTIPLY, i32_kernel_multiply);
    dispatch_register_kernel(DTYPE_I64, CPU, KERNEL_MULTIPLY, i64_kernel_multiply);
    dispatch_register_kernel(DTYPE_U8, CPU, KERNEL_MULTIPLY, u8_kernel_multiply);
    dispatch_register_kernel(DTYPE_U16, CPU, KERNEL_MULTIPLY, u16_kernel_multiply);
    dispatch_register_kernel(DTYPE_U32, CPU, KERNEL_MULTIPLY, u32_kernel_multiply);
    dispatch_register_kernel(DTYPE_U64, CPU, KERNEL_MULTIPLY, u64_kernel_multiply);
    dispatch_register_kernel(DTYPE_F32, CPU, KERNEL_MULTIPLY, f32_kernel_multiply);
    dispatch_register_kernel(DTYPE_F64, CPU, KERNEL_MULTIPLY, f64_kernel_multiply);

    dispatch_register_kernel(DTYPE_I8, CPU, KERNEL_DIVIDE, i8_kernel_divide);
    dispatch_register_kernel(DTYPE_I16, CPU, KERNEL_DIVIDE, i16_kernel_divide);
    dispatch_register_kernel(DTYPE_I32, CPU, KERNEL_DIVIDE, i32_kernel_divide);
    dispatch_register_kernel(DTYPE_I64, CPU, KERNEL_DIVIDE, i64_kernel_divide);
    dispatch_register_kernel(DTYPE_U8, CPU, KERNEL_DIVIDE, u8_kernel_divide);
    dispatch_register_kernel(DTYPE_U16, CPU, KERNEL_DIVIDE, u16_kernel_divide);
    dispatch_register_kernel(DTYPE_U32, CPU, KERNEL_DIVIDE, u32_kernel_divide);
    dispatch_register_kernel(DTYPE_U64, CPU, KERNEL_DIVIDE, u64_kernel_divide);
    dispatch_register_kernel(DTYPE_F32, CPU, KERNEL_DIVIDE, f32_kernel_divide);
    dispatch_register_kernel(DTYPE_F64, CPU, KERNEL_DIVIDE, f64_kernel_divide);

    dispatch_register_kernel(DTYPE_I8, CPU, KERNEL_EQUALITY, i8_kernel_equals);
    dispatch_register_kernel(DTYPE_I16, CPU, KERNEL_EQUALITY, i16_kernel_equals);
    dispatch_register_kernel(DTYPE_I32, CPU, KERNEL_EQUALITY, i32_kernel_equals);
    dispatch_register_kernel(DTYPE_I64, CPU, KERNEL_EQUALITY, i64_kernel_equals);
    dispatch_register_kernel(DTYPE_U8, CPU, KERNEL_EQUALITY, u8_kernel_equals);
    dispatch_register_kernel(DTYPE_U16, CPU, KERNEL_EQUALITY, u16_kernel_equals);
    dispatch_register_kernel(DTYPE_U32, CPU, KERNEL_EQUALITY, u32_kernel_equals);
    dispatch_register_kernel(DTYPE_U64, CPU, KERNEL_EQUALITY, u64_kernel_equals);
    dispatch_register_kernel(DTYPE_F32, CPU, KERNEL_EQUALITY, f32_kernel_equals);
    dispatch_register_kernel(DTYPE_F64, CPU, KERNEL_EQUALITY, f64_kernel_equals);

    dispatch_register_kernel(DTYPE_F32, CPU, KERNEL_NEGATION, f32_kernel_negate);
    dispatch_register_kernel(DTYPE_F64, CPU, KERNEL_NEGATION, f64_kernel_negate);
    dispatch_register_kernel(DTYPE_I16, CPU, KERNEL_NEGATION, i16_kernel_negate);
    dispatch_register_kernel(DTYPE_I32, CPU, KERNEL_NEGATION, i32_kernel_negate);
    dispatch_register_kernel(DTYPE_I64, CPU, KERNEL_NEGATION, i64_kernel_negate);
    dispatch_register_kernel(DTYPE_I8, CPU, KERNEL_NEGATION, i8_kernel_negate);

    dispatch_register_kernel(DTYPE_F32, CPU, KERNEL_ABSOLUTE_VALUE, f32_kernel_abs);
    dispatch_register_kernel(DTYPE_F64, CPU, KERNEL_ABSOLUTE_VALUE, f64_kernel_abs);
    dispatch_register_kernel(DTYPE_I16, CPU, KERNEL_ABSOLUTE_VALUE, i16_kernel_abs);
    dispatch_register_kernel(DTYPE_I32, CPU, KERNEL_ABSOLUTE_VALUE, i32_kernel_abs);
    dispatch_register_kernel(DTYPE_I64, CPU, KERNEL_ABSOLUTE_VALUE, i64_kernel_abs);
    dispatch_register_kernel(DTYPE_I8, CPU, KERNEL_ABSOLUTE_VALUE, i8_kernel_abs);

    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I8_TO_I16, i8_to_i16_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I8_TO_I32, i8_to_i32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I8_TO_I64, i8_to_i64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I8_TO_U8, i8_to_u8_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I8_TO_U16, i8_to_u16_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I8_TO_U32, i8_to_u32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I8_TO_U64, i8_to_u64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I8_TO_F32, i8_to_f32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I8_TO_F64, i8_to_f64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I8_TO_BOOL, i8_to_bool_cast);

    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I16_TO_I8, i16_to_i8_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I16_TO_I32, i16_to_i32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I16_TO_I64, i16_to_i64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I16_TO_U8, i16_to_u8_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I16_TO_U16, i16_to_u16_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I16_TO_U32, i16_to_u32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I16_TO_U64, i16_to_u64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I16_TO_F32, i16_to_f32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I16_TO_F64, i16_to_f64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I16_TO_BOOL, i16_to_bool_cast);

    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I32_TO_I8, i32_to_i8_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I32_TO_I16, i32_to_i16_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I32_TO_I64, i32_to_i64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I32_TO_U8, i32_to_u8_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I32_TO_U16, i32_to_u16_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I32_TO_U32, i32_to_u32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I32_TO_U64, i32_to_u64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I32_TO_F32, i32_to_f32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I32_TO_F64, i32_to_f64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I32_TO_BOOL, i32_to_bool_cast);

    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I64_TO_I8, i64_to_i8_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I64_TO_I16, i64_to_i16_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I64_TO_I32, i64_to_i32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I64_TO_U8, i64_to_u8_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I64_TO_U16, i64_to_u16_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I64_TO_U32, i64_to_u32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I64_TO_U64, i64_to_u64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I64_TO_F32, i64_to_f32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I64_TO_F64, i64_to_f64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_I64_TO_BOOL, i64_to_bool_cast);

    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U8_TO_I8, u8_to_i8_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U8_TO_I16, u8_to_i16_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U8_TO_I32, u8_to_i32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U8_TO_I64, u8_to_i64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U8_TO_U16, u8_to_u16_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U8_TO_U32, u8_to_u32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U8_TO_U64, u8_to_u64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U8_TO_F32, u8_to_f32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U8_TO_F64, u8_to_f64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U8_TO_BOOL, u8_to_bool_cast);

    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U16_TO_I8, u16_to_i8_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U16_TO_I16, u16_to_i16_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U16_TO_I32, u16_to_i32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U16_TO_I64, u16_to_i64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U16_TO_U8, u16_to_u8_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U16_TO_U32, u16_to_u32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U16_TO_U64, u16_to_u64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U16_TO_F32, u16_to_f32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U16_TO_F64, u16_to_f64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U16_TO_BOOL, u16_to_bool_cast);

    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U32_TO_I8, u32_to_i8_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U32_TO_I16, u32_to_i16_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U32_TO_I32, u32_to_i32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U32_TO_I64, u32_to_i64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U32_TO_U8, u32_to_u8_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U32_TO_U16, u32_to_u16_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U32_TO_U64, u32_to_u64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U32_TO_F32, u32_to_f32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U32_TO_F64, u32_to_f64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U32_TO_BOOL, u32_to_bool_cast);

    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U64_TO_I8, u64_to_i8_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U64_TO_I16, u64_to_i16_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U64_TO_I32, u64_to_i32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U64_TO_I64, u64_to_i64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U64_TO_U8, u64_to_u8_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U64_TO_U16, u64_to_u16_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U64_TO_U32, u64_to_u32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U64_TO_F32, u64_to_f32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U64_TO_F64, u64_to_f64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_U64_TO_BOOL, u64_to_bool_cast);

    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_F32_TO_I8, f32_to_i8_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_F32_TO_I16, f32_to_i16_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_F32_TO_I32, f32_to_i32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_F32_TO_I64, f32_to_i64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_F32_TO_U8, f32_to_u8_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_F32_TO_U16, f32_to_u16_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_F32_TO_U32, f32_to_u32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_F32_TO_U64, f32_to_u64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_F32_TO_F64, f32_to_f64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_F32_TO_BOOL, f32_to_bool_cast);

    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_F64_TO_I8, f64_to_i8_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_F64_TO_I16, f64_to_i16_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_F64_TO_I32, f64_to_i32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_F64_TO_I64, f64_to_i64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_F64_TO_U8, f64_to_u8_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_F64_TO_U16, f64_to_u16_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_F64_TO_U32, f64_to_u32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_F64_TO_U64, f64_to_u64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_F64_TO_F32, f64_to_f32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_F64_TO_BOOL, f64_to_bool_cast);

    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_BOOL_TO_I8, bool_to_i8_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_BOOL_TO_I16, bool_to_i16_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_BOOL_TO_I32, bool_to_i32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_BOOL_TO_I64, bool_to_i64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_BOOL_TO_U8, bool_to_u8_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_BOOL_TO_U16, bool_to_u16_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_BOOL_TO_U32, bool_to_u32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_BOOL_TO_U64, bool_to_u64_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_BOOL_TO_F32, bool_to_f32_cast);
    dispatch_register_kernel(DTYPE_NONE, CPU, KERNEL_BOOL_TO_F64, bool_to_f64_cast);


    dispatch_register_kernel(DTYPE_I8, CPU, KERNEL_SUM_REDUCTION, i8_kernel_sum);
    dispatch_register_kernel(DTYPE_I16, CPU, KERNEL_SUM_REDUCTION, i16_kernel_sum);
    dispatch_register_kernel(DTYPE_I32, CPU, KERNEL_SUM_REDUCTION, i32_kernel_sum);
    dispatch_register_kernel(DTYPE_I64, CPU, KERNEL_SUM_REDUCTION, i64_kernel_sum);
    dispatch_register_kernel(DTYPE_U8, CPU, KERNEL_SUM_REDUCTION, u8_kernel_sum);
    dispatch_register_kernel(DTYPE_U16, CPU, KERNEL_SUM_REDUCTION, u16_kernel_sum);
    dispatch_register_kernel(DTYPE_U32, CPU, KERNEL_SUM_REDUCTION, u32_kernel_sum);
    dispatch_register_kernel(DTYPE_U64, CPU, KERNEL_SUM_REDUCTION, u64_kernel_sum);
    dispatch_register_kernel(DTYPE_F32, CPU, KERNEL_SUM_REDUCTION, f32_kernel_sum);
    dispatch_register_kernel(DTYPE_F64, CPU, KERNEL_SUM_REDUCTION, f64_kernel_sum);

    dispatch_register_kernel(DTYPE_I8, CPU, KERNEL_MAX_REDUCTION, i8_kernel_max);
    dispatch_register_kernel(DTYPE_I16, CPU, KERNEL_MAX_REDUCTION, i16_kernel_max);
    dispatch_register_kernel(DTYPE_I32, CPU, KERNEL_MAX_REDUCTION, i32_kernel_max);
    dispatch_register_kernel(DTYPE_I64, CPU, KERNEL_MAX_REDUCTION, i64_kernel_max);
    dispatch_register_kernel(DTYPE_U8, CPU, KERNEL_MAX_REDUCTION, u8_kernel_max);
    dispatch_register_kernel(DTYPE_U16, CPU, KERNEL_MAX_REDUCTION, u16_kernel_max);
    dispatch_register_kernel(DTYPE_U32, CPU, KERNEL_MAX_REDUCTION, u32_kernel_max);
    dispatch_register_kernel(DTYPE_U64, CPU, KERNEL_MAX_REDUCTION, u64_kernel_max);
    dispatch_register_kernel(DTYPE_F32, CPU, KERNEL_MAX_REDUCTION, f32_kernel_max);
    dispatch_register_kernel(DTYPE_F64, CPU, KERNEL_MAX_REDUCTION, f64_kernel_max);

    dispatch_register_kernel(DTYPE_I8, CPU, KERNEL_MIN_REDUCTION, i8_kernel_min);
    dispatch_register_kernel(DTYPE_I16, CPU, KERNEL_MIN_REDUCTION, i16_kernel_min);
    dispatch_register_kernel(DTYPE_I32, CPU, KERNEL_MIN_REDUCTION, i32_kernel_min);
    dispatch_register_kernel(DTYPE_I64, CPU, KERNEL_MIN_REDUCTION, i64_kernel_min);
    dispatch_register_kernel(DTYPE_U8, CPU, KERNEL_MIN_REDUCTION, u8_kernel_min);
    dispatch_register_kernel(DTYPE_U16, CPU, KERNEL_MIN_REDUCTION, u16_kernel_min);
    dispatch_register_kernel(DTYPE_U32, CPU, KERNEL_MIN_REDUCTION, u32_kernel_min);
    dispatch_register_kernel(DTYPE_U64, CPU, KERNEL_MIN_REDUCTION, u64_kernel_min);
    dispatch_register_kernel(DTYPE_F32, CPU, KERNEL_MIN_REDUCTION, f32_kernel_min);
    dispatch_register_kernel(DTYPE_F64, CPU, KERNEL_MIN_REDUCTION, f64_kernel_min);

    dispatch_register_kernel(DTYPE_F32, CPU, KERNEL_MATMUL, f32_kernel_matmul);
    dispatch_register_kernel(DTYPE_F64, CPU, KERNEL_MATMUL, f64_kernel_matmul);


    dispatch_register_kernel(DTYPE_F32, CPU, KERNEL_ADD_BACKWARDS, f32_kernel_add_backward);
    dispatch_register_kernel(DTYPE_F64, CPU, KERNEL_ADD_BACKWARDS, f64_kernel_add_backward);

    dispatch_register_kernel(DTYPE_F32, CPU, KERNEL_SUBTRACT_BACKWARDS, f32_kernel_subtract_backward);
    dispatch_register_kernel(DTYPE_F64, CPU, KERNEL_SUBTRACT_BACKWARDS, f64_kernel_subtract_backward);

    dispatch_register_kernel(DTYPE_F32, CPU, KERNEL_MULTIPLY_BACKWARDS, f32_kernel_multiply_backward);
    dispatch_register_kernel(DTYPE_F64, CPU, KERNEL_MULTIPLY_BACKWARDS, f64_kernel_multiply_backward);

    dispatch_register_kernel(DTYPE_F32, CPU, KERNEL_DIVIDE_BACKWARDS, f32_kernel_divide_backward);
    dispatch_register_kernel(DTYPE_F64, CPU, KERNEL_DIVIDE_BACKWARDS, f64_kernel_divide_backward);

}
