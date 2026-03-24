#include "Kernels.h"


void i8_kernel_add(kernel_info *info) {
    int8_t *in0 = (int8_t*)info->dataptrs[0];
    int8_t *in1 = (int8_t*)info->dataptrs[1];
    int8_t *out = (int8_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 + *in1;
        in0 = (int8_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (int8_t*)((char*)in1 + info->inner_strides[1]);
        out = (int8_t*)((char*)out + info->inner_strides[2]);
    }
}

void i16_kernel_add(kernel_info *info) {
    int16_t *in0 = (int16_t*)info->dataptrs[0];
    int16_t *in1 = (int16_t*)info->dataptrs[1];
    int16_t *out = (int16_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 + *in1;
        in0 = (int16_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (int16_t*)((char*)in1 + info->inner_strides[1]);
        out = (int16_t*)((char*)out + info->inner_strides[2]);
    }
}

void i32_kernel_add(kernel_info *info) {
    int32_t *in0 = (int32_t*)info->dataptrs[0];
    int32_t *in1 = (int32_t*)info->dataptrs[1];
    int32_t *out = (int32_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 + *in1;
        in0 = (int32_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (int32_t*)((char*)in1 + info->inner_strides[1]);
        out = (int32_t*)((char*)out + info->inner_strides[2]);
    }
    
}

void i64_kernel_add(kernel_info *info) {
    int64_t *in0 = (int64_t*)info->dataptrs[0];
    int64_t *in1 = (int64_t*)info->dataptrs[1];
    int64_t *out = (int64_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 + *in1;
        in0 = (int64_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (int64_t*)((char*)in1 + info->inner_strides[1]);
        out = (int64_t*)((char*)out + info->inner_strides[2]);
    }
}

void u8_kernel_add(kernel_info *info) {
    uint8_t *in0 = (uint8_t*)info->dataptrs[0];
    uint8_t *in1 = (uint8_t*)info->dataptrs[1];
    uint8_t *out = (uint8_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 + *in1;
        in0 = (uint8_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (uint8_t*)((char*)in1 + info->inner_strides[1]);
        out = (uint8_t*)((char*)out + info->inner_strides[2]);
    }
}

void u16_kernel_add(kernel_info *info) {
    uint16_t *in0 = (uint16_t*)info->dataptrs[0];
    uint16_t *in1 = (uint16_t*)info->dataptrs[1];
    uint16_t *out = (uint16_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 + *in1;
        in0 = (uint16_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (uint16_t*)((char*)in1 + info->inner_strides[1]);
        out = (uint16_t*)((char*)out + info->inner_strides[2]);
    }
}

void u32_kernel_add(kernel_info *info) {
    uint32_t *in0 = (uint32_t*)info->dataptrs[0];
    uint32_t *in1 = (uint32_t*)info->dataptrs[1];
    uint32_t *out = (uint32_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 + *in1;
        in0 = (uint32_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (uint32_t*)((char*)in1 + info->inner_strides[1]);
        out = (uint32_t*)((char*)out + info->inner_strides[2]);
    }
}

void u64_kernel_add(kernel_info *info) {
    uint64_t *in0 = (uint64_t*)info->dataptrs[0];
    uint64_t *in1 = (uint64_t*)info->dataptrs[1];
    uint64_t *out = (uint64_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 + *in1;
        in0 = (uint64_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (uint64_t*)((char*)in1 + info->inner_strides[1]);
        out = (uint64_t*)((char*)out + info->inner_strides[2]);
    }
}

void f32_kernel_add(kernel_info *info) {
    float *in0 = (float*)info->dataptrs[0];
    float *in1 = (float*)info->dataptrs[1];
    float *out = (float*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 + *in1;
        in0 = (float*)((char*)in0 + info->inner_strides[0]);
        in1 = (float*)((char*)in1 + info->inner_strides[1]);
        out = (float*)((char*)out + info->inner_strides[2]);
    }
}

void f64_kernel_add(kernel_info *info) {
    double *in0 = (double*)info->dataptrs[0];
    double *in1 = (double*)info->dataptrs[1];
    double *out = (double*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 + *in1;
        in0 = (double*)((char*)in0 + info->inner_strides[0]);
        in1 = (double*)((char*)in1 + info->inner_strides[1]);
        out = (double*)((char*)out + info->inner_strides[2]);
    }
}


void i8_kernel_subtract(kernel_info *info) {
    int8_t *in0 = (int8_t*)info->dataptrs[0];
    int8_t *in1 = (int8_t*)info->dataptrs[1];
    int8_t *out = (int8_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 - *in1;
        in0 = (int8_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (int8_t*)((char*)in1 + info->inner_strides[1]);
        out = (int8_t*)((char*)out + info->inner_strides[2]);
    }
}

void i16_kernel_subtract(kernel_info *info) {
    int16_t *in0 = (int16_t*)info->dataptrs[0];
    int16_t *in1 = (int16_t*)info->dataptrs[1];
    int16_t *out = (int16_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 - *in1;
        in0 = (int16_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (int16_t*)((char*)in1 + info->inner_strides[1]);
        out = (int16_t*)((char*)out + info->inner_strides[2]);
    }
}

void i32_kernel_subtract(kernel_info *info) {
    int32_t *in0 = (int32_t*)info->dataptrs[0];
    int32_t *in1 = (int32_t*)info->dataptrs[1];
    int32_t *out = (int32_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 - *in1;
        in0 = (int32_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (int32_t*)((char*)in1 + info->inner_strides[1]);
        out = (int32_t*)((char*)out + info->inner_strides[2]);
    }
}

void i64_kernel_subtract(kernel_info *info) {
    int64_t *in0 = (int64_t*)info->dataptrs[0];
    int64_t *in1 = (int64_t*)info->dataptrs[1];
    int64_t *out = (int64_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 - *in1;
        in0 = (int64_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (int64_t*)((char*)in1 + info->inner_strides[1]);
        out = (int64_t*)((char*)out + info->inner_strides[2]);
    }
}

void u8_kernel_subtract(kernel_info *info) {
    uint8_t *in0 = (uint8_t*)info->dataptrs[0];
    uint8_t *in1 = (uint8_t*)info->dataptrs[1];
    uint8_t *out = (uint8_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 - *in1;
        in0 = (uint8_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (uint8_t*)((char*)in1 + info->inner_strides[1]);
        out = (uint8_t*)((char*)out + info->inner_strides[2]);
    }
}

void u16_kernel_subtract(kernel_info *info) {
    uint16_t *in0 = (uint16_t*)info->dataptrs[0];
    uint16_t *in1 = (uint16_t*)info->dataptrs[1];
    uint16_t *out = (uint16_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 - *in1;
        in0 = (uint16_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (uint16_t*)((char*)in1 + info->inner_strides[1]);
        out = (uint16_t*)((char*)out + info->inner_strides[2]);
    }
}

void u32_kernel_subtract(kernel_info *info) {
    uint32_t *in0 = (uint32_t*)info->dataptrs[0];
    uint32_t *in1 = (uint32_t*)info->dataptrs[1];
    uint32_t *out = (uint32_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 - *in1;
        in0 = (uint32_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (uint32_t*)((char*)in1 + info->inner_strides[1]);
        out = (uint32_t*)((char*)out + info->inner_strides[2]);
    }
}

void u64_kernel_subtract(kernel_info *info) {
    uint64_t *in0 = (uint64_t*)info->dataptrs[0];
    uint64_t *in1 = (uint64_t*)info->dataptrs[1];
    uint64_t *out = (uint64_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 - *in1;
        in0 = (uint64_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (uint64_t*)((char*)in1 + info->inner_strides[1]);
        out = (uint64_t*)((char*)out + info->inner_strides[2]);
    }
}

void f32_kernel_subtract(kernel_info *info) {
    float *in0 = (float*)info->dataptrs[0];
    float *in1 = (float*)info->dataptrs[1];
    float *out = (float*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 - *in1;
        in0 = (float*)((char*)in0 + info->inner_strides[0]);
        in1 = (float*)((char*)in1 + info->inner_strides[1]);
        out = (float*)((char*)out + info->inner_strides[2]);
    }
}

void f64_kernel_subtract(kernel_info *info) {
    double *in0 = (double*)info->dataptrs[0];
    double *in1 = (double*)info->dataptrs[1];
    double *out = (double*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 - *in1;
        in0 = (double*)((char*)in0 + info->inner_strides[0]);
        in1 = (double*)((char*)in1 + info->inner_strides[1]);
        out = (double*)((char*)out + info->inner_strides[2]);
    }
}


void i8_kernel_multiply(kernel_info *info) {
    int8_t *in0 = (int8_t*)info->dataptrs[0];
    int8_t *in1 = (int8_t*)info->dataptrs[1];
    int8_t *out = (int8_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 * *in1;
        in0 = (int8_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (int8_t*)((char*)in1 + info->inner_strides[1]);
        out = (int8_t*)((char*)out + info->inner_strides[2]);
    }
}

void i16_kernel_multiply(kernel_info *info) {
    int16_t *in0 = (int16_t*)info->dataptrs[0];
    int16_t *in1 = (int16_t*)info->dataptrs[1];
    int16_t *out = (int16_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 * *in1;
        in0 = (int16_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (int16_t*)((char*)in1 + info->inner_strides[1]);
        out = (int16_t*)((char*)out + info->inner_strides[2]);
    }
}

void i32_kernel_multiply(kernel_info *info) {
    int32_t *in0 = (int32_t*)info->dataptrs[0];
    int32_t *in1 = (int32_t*)info->dataptrs[1];
    int32_t *out = (int32_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 * *in1;
        in0 = (int32_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (int32_t*)((char*)in1 + info->inner_strides[1]);
        out = (int32_t*)((char*)out + info->inner_strides[2]);
    }
}

void i64_kernel_multiply(kernel_info *info) {
    int64_t *in0 = (int64_t*)info->dataptrs[0];
    int64_t *in1 = (int64_t*)info->dataptrs[1];
    int64_t *out = (int64_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 * *in1;
        in0 = (int64_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (int64_t*)((char*)in1 + info->inner_strides[1]);
        out = (int64_t*)((char*)out + info->inner_strides[2]);
    }
}

void u8_kernel_multiply(kernel_info *info) {
    uint8_t *in0 = (uint8_t*)info->dataptrs[0];
    uint8_t *in1 = (uint8_t*)info->dataptrs[1];
    uint8_t *out = (uint8_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 * *in1;
        in0 = (uint8_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (uint8_t*)((char*)in1 + info->inner_strides[1]);
        out = (uint8_t*)((char*)out + info->inner_strides[2]);
    }
}

void u16_kernel_multiply(kernel_info *info) {
    uint16_t *in0 = (uint16_t*)info->dataptrs[0];
    uint16_t *in1 = (uint16_t*)info->dataptrs[1];
    uint16_t *out = (uint16_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 * *in1;
        in0 = (uint16_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (uint16_t*)((char*)in1 + info->inner_strides[1]);
        out = (uint16_t*)((char*)out + info->inner_strides[2]);
    }
}

void u32_kernel_multiply(kernel_info *info) {
    uint32_t *in0 = (uint32_t*)info->dataptrs[0];
    uint32_t *in1 = (uint32_t*)info->dataptrs[1];
    uint32_t *out = (uint32_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 * *in1;
        in0 = (uint32_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (uint32_t*)((char*)in1 + info->inner_strides[1]);
        out = (uint32_t*)((char*)out + info->inner_strides[2]);
    }
}

void u64_kernel_multiply(kernel_info *info) {
    uint64_t *in0 = (uint64_t*)info->dataptrs[0];
    uint64_t *in1 = (uint64_t*)info->dataptrs[1];
    uint64_t *out = (uint64_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 * *in1;
        in0 = (uint64_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (uint64_t*)((char*)in1 + info->inner_strides[1]);
        out = (uint64_t*)((char*)out + info->inner_strides[2]);
    }
}

void f32_kernel_multiply(kernel_info *info) {
    float *in0 = (float*)info->dataptrs[0];
    float *in1 = (float*)info->dataptrs[1];
    float *out = (float*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 * *in1;
        in0 = (float*)((char*)in0 + info->inner_strides[0]);
        in1 = (float*)((char*)in1 + info->inner_strides[1]);
        out = (float*)((char*)out + info->inner_strides[2]);
    }
}

void f64_kernel_multiply(kernel_info *info) {
    double *in0 = (double*)info->dataptrs[0];
    double *in1 = (double*)info->dataptrs[1];
    double *out = (double*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 * *in1;
        in0 = (double*)((char*)in0 + info->inner_strides[0]);
        in1 = (double*)((char*)in1 + info->inner_strides[1]);
        out = (double*)((char*)out + info->inner_strides[2]);
    }
}


void i8_kernel_divide(kernel_info *info) {
    int8_t *in0 = (int8_t*)info->dataptrs[0];
    int8_t *in1 = (int8_t*)info->dataptrs[1];
    int8_t *out = (int8_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 / *in1;
        in0 = (int8_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (int8_t*)((char*)in1 + info->inner_strides[1]);
        out = (int8_t*)((char*)out + info->inner_strides[2]);
    }
}

void i16_kernel_divide(kernel_info *info) {
    int16_t *in0 = (int16_t*)info->dataptrs[0];
    int16_t *in1 = (int16_t*)info->dataptrs[1];
    int16_t *out = (int16_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 / *in1;
        in0 = (int16_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (int16_t*)((char*)in1 + info->inner_strides[1]);
        out = (int16_t*)((char*)out + info->inner_strides[2]);
    }
}

void i32_kernel_divide(kernel_info *info) {
    int32_t *in0 = (int32_t*)info->dataptrs[0];
    int32_t *in1 = (int32_t*)info->dataptrs[1];
    int32_t *out = (int32_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 / *in1;
        in0 = (int32_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (int32_t*)((char*)in1 + info->inner_strides[1]);
        out = (int32_t*)((char*)out + info->inner_strides[2]);
    }
}

void i64_kernel_divide(kernel_info *info) {
    int64_t *in0 = (int64_t*)info->dataptrs[0];
    int64_t *in1 = (int64_t*)info->dataptrs[1];
    int64_t *out = (int64_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 / *in1;
        in0 = (int64_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (int64_t*)((char*)in1 + info->inner_strides[1]);
        out = (int64_t*)((char*)out + info->inner_strides[2]);
    }
}

void u8_kernel_divide(kernel_info *info) {
    uint8_t *in0 = (uint8_t*)info->dataptrs[0];
    uint8_t *in1 = (uint8_t*)info->dataptrs[1];
    uint8_t *out = (uint8_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 / *in1;
        in0 = (uint8_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (uint8_t*)((char*)in1 + info->inner_strides[1]);
        out = (uint8_t*)((char*)out + info->inner_strides[2]);
    }
}

void u16_kernel_divide(kernel_info *info) {
    uint16_t *in0 = (uint16_t*)info->dataptrs[0];
    uint16_t *in1 = (uint16_t*)info->dataptrs[1];
    uint16_t *out = (uint16_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 / *in1;
        in0 = (uint16_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (uint16_t*)((char*)in1 + info->inner_strides[1]);
        out = (uint16_t*)((char*)out + info->inner_strides[2]);
    }
}

void u32_kernel_divide(kernel_info *info) {
    uint32_t *in0 = (uint32_t*)info->dataptrs[0];
    uint32_t *in1 = (uint32_t*)info->dataptrs[1];
    uint32_t *out = (uint32_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 / *in1;
        in0 = (uint32_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (uint32_t*)((char*)in1 + info->inner_strides[1]);
        out = (uint32_t*)((char*)out + info->inner_strides[2]);
    }
}

void u64_kernel_divide(kernel_info *info) {
    uint64_t *in0 = (uint64_t*)info->dataptrs[0];
    uint64_t *in1 = (uint64_t*)info->dataptrs[1];
    uint64_t *out = (uint64_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 / *in1;
        in0 = (uint64_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (uint64_t*)((char*)in1 + info->inner_strides[1]);
        out = (uint64_t*)((char*)out + info->inner_strides[2]);
    }
}

void f32_kernel_divide(kernel_info *info) {
    float *in0 = (float*)info->dataptrs[0];
    float *in1 = (float*)info->dataptrs[1];
    float *out = (float*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 / *in1;
        in0 = (float*)((char*)in0 + info->inner_strides[0]);
        in1 = (float*)((char*)in1 + info->inner_strides[1]);
        out = (float*)((char*)out + info->inner_strides[2]);
    }
}

void f64_kernel_divide(kernel_info *info) {
    double *in0 = (double*)info->dataptrs[0];
    double *in1 = (double*)info->dataptrs[1];
    double *out = (double*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 / *in1;
        in0 = (double*)((char*)in0 + info->inner_strides[0]);
        in1 = (double*)((char*)in1 + info->inner_strides[1]);
        out = (double*)((char*)out + info->inner_strides[2]);
    }
}


void i8_kernel_equals(kernel_info *info) {
    int8_t *in0 = (int8_t*)info->dataptrs[0];
    int8_t *in1 = (int8_t*)info->dataptrs[1];
    int8_t *out = (int8_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 == *in1;
        in0 = (int8_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (int8_t*)((char*)in1 + info->inner_strides[1]);
        out = (int8_t*)((char*)out + info->inner_strides[2]);
    }
}

void i16_kernel_equals(kernel_info *info) {
    int16_t *in0 = (int16_t*)info->dataptrs[0];
    int16_t *in1 = (int16_t*)info->dataptrs[1];
    int16_t *out = (int16_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 == *in1;
        in0 = (int16_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (int16_t*)((char*)in1 + info->inner_strides[1]);
        out = (int16_t*)((char*)out + info->inner_strides[2]);
    }
}

void i32_kernel_equals(kernel_info *info) {
    int32_t *in0 = (int32_t*)info->dataptrs[0];
    int32_t *in1 = (int32_t*)info->dataptrs[1];
    int32_t *out = (int32_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 == *in1;
        in0 = (int32_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (int32_t*)((char*)in1 + info->inner_strides[1]);
        out = (int32_t*)((char*)out + info->inner_strides[2]);
    }
}

void i64_kernel_equals(kernel_info *info) {
    int64_t *in0 = (int64_t*)info->dataptrs[0];
    int64_t *in1 = (int64_t*)info->dataptrs[1];
    int64_t *out = (int64_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 == *in1;
        in0 = (int64_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (int64_t*)((char*)in1 + info->inner_strides[1]);
        out = (int64_t*)((char*)out + info->inner_strides[2]);
    }
}

void u8_kernel_equals(kernel_info *info) {
    uint8_t *in0 = (uint8_t*)info->dataptrs[0];
    uint8_t *in1 = (uint8_t*)info->dataptrs[1];
    uint8_t *out = (uint8_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 == *in1;
        in0 = (uint8_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (uint8_t*)((char*)in1 + info->inner_strides[1]);
        out = (uint8_t*)((char*)out + info->inner_strides[2]);
    }
}

void u16_kernel_equals(kernel_info *info) {
    uint16_t *in0 = (uint16_t*)info->dataptrs[0];
    uint16_t *in1 = (uint16_t*)info->dataptrs[1];
    uint16_t *out = (uint16_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 == *in1;
        in0 = (uint16_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (uint16_t*)((char*)in1 + info->inner_strides[1]);
        out = (uint16_t*)((char*)out + info->inner_strides[2]);
    }
}

void u32_kernel_equals(kernel_info *info) {
    uint32_t *in0 = (uint32_t*)info->dataptrs[0];
    uint32_t *in1 = (uint32_t*)info->dataptrs[1];
    uint32_t *out = (uint32_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 == *in1;
        in0 = (uint32_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (uint32_t*)((char*)in1 + info->inner_strides[1]);
        out = (uint32_t*)((char*)out + info->inner_strides[2]);
    }
}

void u64_kernel_equals(kernel_info *info) {
    uint64_t *in0 = (uint64_t*)info->dataptrs[0];
    uint64_t *in1 = (uint64_t*)info->dataptrs[1];
    uint64_t *out = (uint64_t*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 == *in1;
        in0 = (uint64_t*)((char*)in0 + info->inner_strides[0]);
        in1 = (uint64_t*)((char*)in1 + info->inner_strides[1]);
        out = (uint64_t*)((char*)out + info->inner_strides[2]);
    }
}

void f32_kernel_equals(kernel_info *info) {
    float *in0 = (float*)info->dataptrs[0];
    float *in1 = (float*)info->dataptrs[1];
    float *out = (float*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 == *in1;
        in0 = (float*)((char*)in0 + info->inner_strides[0]);
        in1 = (float*)((char*)in1 + info->inner_strides[1]);
        out = (float*)((char*)out + info->inner_strides[2]);
    }
}

void f64_kernel_equals(kernel_info *info) {
    double *in0 = (double*)info->dataptrs[0];
    double *in1 = (double*)info->dataptrs[1];
    double *out = (double*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = *in0 == *in1;
        in0 = (double*)((char*)in0 + info->inner_strides[0]);
        in1 = (double*)((char*)in1 + info->inner_strides[1]);
        out = (double*)((char*)out + info->inner_strides[2]);
    }
}


void f32_kernel_negate(kernel_info *info) {
    float *in0 = (float*)info->dataptrs[0];
    float *out = (float*)info->dataptrs[1];
    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = -(*in0);
        in0 = (float*)((char*)in0 + info->inner_strides[0]);
        out = (float*)((char*)out + info->inner_strides[1]);
    }
}

void f64_kernel_negate(kernel_info *info) {
    double *in0 = (double*)info->dataptrs[0];
    double *out = (double*)info->dataptrs[1];
    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = -(*in0);
        in0 = (double*)((char*)in0 + info->inner_strides[0]);
        out = (double*)((char*)out + info->inner_strides[1]);
    }
}

void i16_kernel_negate(kernel_info *info) {
    int16_t *in0 = (int16_t*)info->dataptrs[0];
    int16_t *out = (int16_t*)info->dataptrs[1];
    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = -(*in0);
        in0 = (int16_t*)((char*)in0 + info->inner_strides[0]);
        out = (int16_t*)((char*)out + info->inner_strides[1]);
    }
}

void i32_kernel_negate(kernel_info *info) {
    int32_t *in0 = (int32_t*)info->dataptrs[0];
    int32_t *out = (int32_t*)info->dataptrs[1];
    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = -(*in0);
        in0 = (int32_t*)((char*)in0 + info->inner_strides[0]);
        out = (int32_t*)((char*)out + info->inner_strides[1]);
    }
}

void i64_kernel_negate(kernel_info *info) {
    int64_t *in0 = (int64_t*)info->dataptrs[0];
    int64_t *out = (int64_t*)info->dataptrs[1];
    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = -(*in0);
        in0 = (int64_t*)((char*)in0 + info->inner_strides[0]);
        out = (int64_t*)((char*)out + info->inner_strides[1]);
    }
}

void i8_kernel_negate(kernel_info *info) {
    int8_t *in0 = (int8_t*)info->dataptrs[0];
    int8_t *out = (int8_t*)info->dataptrs[1];
    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = -(*in0);
        in0 = (int8_t*)((char*)in0 + info->inner_strides[0]);
        out = (int8_t*)((char*)out + info->inner_strides[1]);
    }
}


void f32_kernel_abs(kernel_info *info) {
    float *in0 = (float*)info->dataptrs[0];
    float *out = (float*)info->dataptrs[1];
    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = fabs(*in0);
        in0 = (float*)((char*)in0 + info->inner_strides[0]);
        out = (float*)((char*)out + info->inner_strides[1]);
    }
}

void f64_kernel_abs(kernel_info *info) {
    double *in0 = (double*)info->dataptrs[0];
    double *out = (double*)info->dataptrs[1];
    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = fabs(*in0);
        in0 = (double*)((char*)in0 + info->inner_strides[0]);
        out = (double*)((char*)out + info->inner_strides[1]);
    }
}

void i16_kernel_abs(kernel_info *info) {
    int16_t *in0 = (int16_t*)info->dataptrs[0];
    int16_t *out = (int16_t*)info->dataptrs[1];
    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (*in0 < 0) ? -*in0 : *in0;
        in0 = (int16_t*)((char*)in0 + info->inner_strides[0]);
        out = (int16_t*)((char*)out + info->inner_strides[1]);
    }
}

void i32_kernel_abs(kernel_info *info) {
    int32_t *in0 = (int32_t*)info->dataptrs[0];
    int32_t *out = (int32_t*)info->dataptrs[1];
    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (*in0 < 0) ? -*in0 : *in0;
        in0 = (int32_t*)((char*)in0 + info->inner_strides[0]);
        out = (int32_t*)((char*)out + info->inner_strides[1]);
    }
}

void i64_kernel_abs(kernel_info *info) {
    int64_t *in0 = (int64_t*)info->dataptrs[0];
    int64_t *out = (int64_t*)info->dataptrs[1];
    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (*in0 < 0) ? -*in0 : *in0;
        in0 = (int64_t*)((char*)in0 + info->inner_strides[0]);
        out = (int64_t*)((char*)out + info->inner_strides[1]);
    }
}

void i8_kernel_abs(kernel_info *info) {
    int8_t *in0 = (int8_t*)info->dataptrs[0];
    int8_t *out = (int8_t*)info->dataptrs[1];
    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (*in0 < 0) ? -*in0 : *in0;
        in0 = (int8_t*)((char*)in0 + info->inner_strides[0]);
        out = (int8_t*)((char*)out + info->inner_strides[1]);
    }
}


void i8_kernel_sum(kernel_info *info) {
    int R = info->rndims;
    int8_t *out = (int8_t*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        int8_t in0 = *(int8_t*)in;

        *out += in0;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void i16_kernel_sum(kernel_info *info) {
    int R = info->rndims;
    int16_t *out = (int16_t*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        int16_t in0 = *(int16_t*)in;

        *out += in0;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void i32_kernel_sum(kernel_info *info) {
    int R = info->rndims;
    int32_t *out = (int32_t*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        int32_t in0 = *(int32_t*)in;

        *out += in0;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void i64_kernel_sum(kernel_info *info) {
    int R = info->rndims;
    int64_t *out = (int64_t*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        int64_t in0 = *(int64_t*)in;

        *out += in0;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void u8_kernel_sum(kernel_info *info) {
    int R = info->rndims;
    uint8_t *out = (uint8_t*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        uint8_t in0 = *(uint8_t*)in;

        *out += in0;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void u16_kernel_sum(kernel_info *info) {
    int R = info->rndims;
    uint16_t *out = (uint16_t*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        uint16_t in0 = *(uint16_t*)in;

        *out += in0;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void u32_kernel_sum(kernel_info *info) {
    int R = info->rndims;
    uint32_t *out = (uint32_t*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        uint32_t in0 = *(uint32_t*)in;

        *out += in0;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void u64_kernel_sum(kernel_info *info) {
    int R = info->rndims;
    uint64_t *out = (uint64_t*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        uint64_t in0 = *(uint64_t*)in;

        *out += in0;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void f32_kernel_sum(kernel_info *info) {
    int R = info->rndims;
    float *out = (float*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        float in0 = *(float*)in;

        *out += in0;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void f64_kernel_sum(kernel_info *info) {
    int R = info->rndims;
    double *out = (double*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        double in0 = *(double*)in;

        *out += in0;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}


void i8_kernel_max(kernel_info *info) {
    int R = info->rndims;
    int8_t *out = (int8_t*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        int8_t in0 = *(int8_t*)in;

        int8_t out0 = *(int8_t*)out;
		*out = in0 > out0 ? in0 : out0;;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void i16_kernel_max(kernel_info *info) {
    int R = info->rndims;
    int16_t *out = (int16_t*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        int16_t in0 = *(int16_t*)in;

        int16_t out0 = *(int16_t*)out;
		*out = in0 > out0 ? in0 : out0;;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void i32_kernel_max(kernel_info *info) {
    int R = info->rndims;
    int32_t *out = (int32_t*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        int32_t in0 = *(int32_t*)in;

        int32_t out0 = *(int32_t*)out;
		*out = in0 > out0 ? in0 : out0;;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void i64_kernel_max(kernel_info *info) {
    int R = info->rndims;
    int64_t *out = (int64_t*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        int64_t in0 = *(int64_t*)in;

        int64_t out0 = *(int64_t*)out;
		*out = in0 > out0 ? in0 : out0;;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void u8_kernel_max(kernel_info *info) {
    int R = info->rndims;
    uint8_t *out = (uint8_t*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        uint8_t in0 = *(uint8_t*)in;

        uint8_t out0 = *(uint8_t*)out;
		*out = in0 > out0 ? in0 : out0;;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void u16_kernel_max(kernel_info *info) {
    int R = info->rndims;
    uint16_t *out = (uint16_t*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        uint16_t in0 = *(uint16_t*)in;

        uint16_t out0 = *(uint16_t*)out;
		*out = in0 > out0 ? in0 : out0;;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void u32_kernel_max(kernel_info *info) {
    int R = info->rndims;
    uint32_t *out = (uint32_t*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        uint32_t in0 = *(uint32_t*)in;

        uint32_t out0 = *(uint32_t*)out;
		*out = in0 > out0 ? in0 : out0;;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void u64_kernel_max(kernel_info *info) {
    int R = info->rndims;
    uint64_t *out = (uint64_t*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        uint64_t in0 = *(uint64_t*)in;

        uint64_t out0 = *(uint64_t*)out;
		*out = in0 > out0 ? in0 : out0;;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void f32_kernel_max(kernel_info *info) {
    int R = info->rndims;
    float *out = (float*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        float in0 = *(float*)in;

        float out0 = *(float*)out;
		*out = in0 > out0 ? in0 : out0;;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void f64_kernel_max(kernel_info *info) {
    int R = info->rndims;
    double *out = (double*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        double in0 = *(double*)in;

        double out0 = *(double*)out;
		*out = in0 > out0 ? in0 : out0;;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}


void i8_kernel_min(kernel_info *info) {
    int R = info->rndims;
    int8_t *out = (int8_t*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        int8_t in0 = *(int8_t*)in;

        int8_t out0 = *(int8_t*)out;
		*out = in0 < out0 ? in0 : out0;;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void i16_kernel_min(kernel_info *info) {
    int R = info->rndims;
    int16_t *out = (int16_t*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        int16_t in0 = *(int16_t*)in;

        int16_t out0 = *(int16_t*)out;
		*out = in0 < out0 ? in0 : out0;;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void i32_kernel_min(kernel_info *info) {
    int R = info->rndims;
    int32_t *out = (int32_t*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        int32_t in0 = *(int32_t*)in;

        int32_t out0 = *(int32_t*)out;
		*out = in0 < out0 ? in0 : out0;;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void i64_kernel_min(kernel_info *info) {
    int R = info->rndims;
    int64_t *out = (int64_t*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        int64_t in0 = *(int64_t*)in;

        int64_t out0 = *(int64_t*)out;
		*out = in0 < out0 ? in0 : out0;;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void u8_kernel_min(kernel_info *info) {
    int R = info->rndims;
    uint8_t *out = (uint8_t*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        uint8_t in0 = *(uint8_t*)in;

        uint8_t out0 = *(uint8_t*)out;
		*out = in0 < out0 ? in0 : out0;;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void u16_kernel_min(kernel_info *info) {
    int R = info->rndims;
    uint16_t *out = (uint16_t*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        uint16_t in0 = *(uint16_t*)in;

        uint16_t out0 = *(uint16_t*)out;
		*out = in0 < out0 ? in0 : out0;;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void u32_kernel_min(kernel_info *info) {
    int R = info->rndims;
    uint32_t *out = (uint32_t*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        uint32_t in0 = *(uint32_t*)in;

        uint32_t out0 = *(uint32_t*)out;
		*out = in0 < out0 ? in0 : out0;;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void u64_kernel_min(kernel_info *info) {
    int R = info->rndims;
    uint64_t *out = (uint64_t*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        uint64_t in0 = *(uint64_t*)in;

        uint64_t out0 = *(uint64_t*)out;
		*out = in0 < out0 ? in0 : out0;;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void f32_kernel_min(kernel_info *info) {
    int R = info->rndims;
    float *out = (float*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        float in0 = *(float*)in;

        float out0 = *(float*)out;
		*out = in0 < out0 ? in0 : out0;;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void f64_kernel_min(kernel_info *info) {
    int R = info->rndims;
    double *out = (double*)info->dataptrs[1];
    
    int64_t idx[16] = {0}; 
    if (R > 16) return; 

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) {
            in += idx[d] * info->reduce_strides[0][d];
        }
        
        double in0 = *(double*)in;

        double out0 = *(double*)out;
		*out = in0 < out0 ? in0 : out0;;

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}


void f32_kernel_matmul(kernel_info *info) {
    float *in0 = (float*)info->dataptrs[0];
    float *in1 = (float*)info->dataptrs[1];
    float *out = (float*)info->dataptrs[2];
    int M = info->arr_a_row_mm;
    int K = info->arr_a_col_mm;
    int N = info->arr_b_col_mm;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, in0, K, in1, N, 0, out, N);
    
}

void f64_kernel_matmul(kernel_info *info) {
    double *in0 = (double*)info->dataptrs[0];
    double *in1 = (double*)info->dataptrs[1];
    double *out = (double*)info->dataptrs[2];
    int M = info->arr_a_row_mm;
    int K = info->arr_a_col_mm;
    int N = info->arr_b_col_mm;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, in0, K, in1, N, 0, out, N);
    
}





void i8_kernel_argmax(kernel_info *info) {
    int R = info->rndims;
    int64_t *out = (int64_t*)info->dataptrs[1];
    int8_t best_val = *(int8_t*)info->dataptrs[2]; 
    
    int64_t idx[16] = {0};
    if (R > 16) return;

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) in += idx[d] * info->reduce_strides[0][d];
        int8_t in0 = *(int8_t*)in;

        int64_t current_flat_idx = 0;
        for (int d = 0; d < R; d++) current_flat_idx += idx[d] * info->flat_strides[d];

        int8_t out0 = *(int8_t*)out;
		if (in0 > best_val) { best_val = in0; *out = current_flat_idx; }

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void i16_kernel_argmax(kernel_info *info) {
    int R = info->rndims;
    int64_t *out = (int64_t*)info->dataptrs[1];
    int16_t best_val = *(int16_t*)info->dataptrs[2]; 
    
    int64_t idx[16] = {0};
    if (R > 16) return;

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) in += idx[d] * info->reduce_strides[0][d];
        int16_t in0 = *(int16_t*)in;

        int64_t current_flat_idx = 0;
        for (int d = 0; d < R; d++) current_flat_idx += idx[d] * info->flat_strides[d];

        int16_t out0 = *(int16_t*)out;
		if (in0 > best_val) { best_val = in0; *out = current_flat_idx; }

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void i32_kernel_argmax(kernel_info *info) {
    int R = info->rndims;
    int64_t *out = (int64_t*)info->dataptrs[1];
    int32_t best_val = *(int32_t*)info->dataptrs[2]; 
    
    int64_t idx[16] = {0};
    if (R > 16) return;

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) in += idx[d] * info->reduce_strides[0][d];
        int32_t in0 = *(int32_t*)in;

        int64_t current_flat_idx = 0;
        for (int d = 0; d < R; d++) current_flat_idx += idx[d] * info->flat_strides[d];

        int32_t out0 = *(int32_t*)out;
		if (in0 > best_val) { best_val = in0; *out = current_flat_idx; }

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void i64_kernel_argmax(kernel_info *info) {
    int R = info->rndims;
    int64_t *in_base = (int64_t*)info->dataptrs[0];
    int64_t *out_idx = (int64_t*)info->dataptrs[1]; // Output is indices
    
    // We need a place to store the best value found so far for this chunk
    // You can either use a temporary variable or a 3rd data pointer
    int64_t max_val = INT64_MAX; // Identity for int8 max
    int64_t best_idx = 0;

    int64_t idx[16] = {0};
    if (R > 16) return;

    while (1) {
        // 1. Get input value
        char *in_ptr = (char*)in_base;
        for (int d = 0; d < R; d++) {
            in_ptr += idx[d] * info->reduce_strides[0][d];
        }
        int64_t val = *(int64_t*)in_ptr;

        // 2. Convert N-D idx to flat block-local index
        int64_t current_flat_idx = 0;
        for (int d = 0; d < R; d++) {
            current_flat_idx += idx[d] * info->flat_strides[d];
        }

        // 3. Compare and Update
        if (val > max_val) {
            max_val = val;
            best_idx = current_flat_idx;
        }

        // Standard N-D Increment
        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
    
    *out_idx = best_idx;
}

void u8_kernel_argmax(kernel_info *info) {
    int R = info->rndims;
    int64_t *out = (int64_t*)info->dataptrs[1];
    uint8_t best_val = *(uint8_t*)info->dataptrs[2]; 
    
    int64_t idx[16] = {0};
    if (R > 16) return;

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) in += idx[d] * info->reduce_strides[0][d];
        uint8_t in0 = *(uint8_t*)in;

        int64_t current_flat_idx = 0;
        for (int d = 0; d < R; d++) current_flat_idx += idx[d] * info->flat_strides[d];

        uint8_t out0 = *(uint8_t*)out;
		if (in0 > best_val) { best_val = in0; *out = current_flat_idx; }

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void u16_kernel_argmax(kernel_info *info) {
    int R = info->rndims;
    int64_t *out = (int64_t*)info->dataptrs[1];
    uint16_t best_val = *(uint16_t*)info->dataptrs[2]; 
    
    int64_t idx[16] = {0};
    if (R > 16) return;

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) in += idx[d] * info->reduce_strides[0][d];
        uint16_t in0 = *(uint16_t*)in;

        int64_t current_flat_idx = 0;
        for (int d = 0; d < R; d++) current_flat_idx += idx[d] * info->flat_strides[d];

        uint16_t out0 = *(uint16_t*)out;
		if (in0 > best_val) { best_val = in0; *out = current_flat_idx; }

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void u32_kernel_argmax(kernel_info *info) {
    int R = info->rndims;
    int64_t *out = (int64_t*)info->dataptrs[1];
    uint32_t best_val = *(uint32_t*)info->dataptrs[2]; 
    
    int64_t idx[16] = {0};
    if (R > 16) return;

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) in += idx[d] * info->reduce_strides[0][d];
        uint32_t in0 = *(uint32_t*)in;

        int64_t current_flat_idx = 0;
        for (int d = 0; d < R; d++) current_flat_idx += idx[d] * info->flat_strides[d];

        uint32_t out0 = *(uint32_t*)out;
		if (in0 > best_val) { best_val = in0; *out = current_flat_idx; }

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void u64_kernel_argmax(kernel_info *info) {
    int R = info->rndims;
    int64_t *out = (int64_t*)info->dataptrs[1];
    uint64_t best_val = *(uint64_t*)info->dataptrs[2]; 
    
    int64_t idx[16] = {0};
    if (R > 16) return;

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) in += idx[d] * info->reduce_strides[0][d];
        uint64_t in0 = *(uint64_t*)in;

        int64_t current_flat_idx = 0;
        for (int d = 0; d < R; d++) current_flat_idx += idx[d] * info->flat_strides[d];

        uint64_t out0 = *(uint64_t*)out;
		if (in0 > best_val) { best_val = in0; *out = current_flat_idx; }

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void f32_kernel_argmax(kernel_info *info) {
    int R = info->rndims;
    int64_t *out = (int64_t*)info->dataptrs[1];
    float best_val = *(float*)info->dataptrs[2]; 
    
    int64_t idx[16] = {0};
    if (R > 16) return;

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) in += idx[d] * info->reduce_strides[0][d];
        float in0 = *(float*)in;

        int64_t current_flat_idx = 0;
        for (int d = 0; d < R; d++) current_flat_idx += idx[d] * info->flat_strides[d];

        float out0 = *(float*)out;
		if (in0 > best_val) { best_val = in0; *out = current_flat_idx; }

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}

void f64_kernel_argmax(kernel_info *info) {
    int R = info->rndims;
    int64_t *out = (int64_t*)info->dataptrs[1];
    double best_val = *(double*)info->dataptrs[2]; 
    
    int64_t idx[16] = {0};
    if (R > 16) return;

    while (1) {
        char *in = info->dataptrs[0];
        for (int d = 0; d < R; d++) in += idx[d] * info->reduce_strides[0][d];
        double in0 = *(double*)in;

        int64_t current_flat_idx = 0;
        for (int d = 0; d < R; d++) current_flat_idx += idx[d] * info->flat_strides[d];

        double out0 = *(double*)out;
		if (in0 > best_val) { best_val = in0; *out = current_flat_idx; }

        int d = R - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < info->reduce_shape[d]) break;
            idx[d] = 0;
            d--;
        }
        if (d < 0) break;
    }
}


void i8_to_i16_cast(kernel_info *info) {
    int8_t *in0 = (int8_t*)info->dataptrs[0];
    int16_t *out = (int16_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int16_t)(*in0);
         in0 = (int8_t*)((char*)in0 + info->inner_strides[0]);
         out = (int16_t*)((char*)out + info->inner_strides[1]);
    }
}

void i8_to_i32_cast(kernel_info *info) {
    int8_t *in0 = (int8_t*)info->dataptrs[0];
    int32_t *out = (int32_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int32_t)(*in0);
         in0 = (int8_t*)((char*)in0 + info->inner_strides[0]);
         out = (int32_t*)((char*)out + info->inner_strides[1]);
    }
}

void i8_to_i64_cast(kernel_info *info) {
    int8_t *in0 = (int8_t*)info->dataptrs[0];
    int64_t *out = (int64_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int64_t)(*in0);
         in0 = (int8_t*)((char*)in0 + info->inner_strides[0]);
         out = (int64_t*)((char*)out + info->inner_strides[1]);
    }
}

void i8_to_u8_cast(kernel_info *info) {
    int8_t *in0 = (int8_t*)info->dataptrs[0];
    uint8_t *out = (uint8_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint8_t)(*in0);
         in0 = (int8_t*)((char*)in0 + info->inner_strides[0]);
         out = (uint8_t*)((char*)out + info->inner_strides[1]);
    }
}

void i8_to_u16_cast(kernel_info *info) {
    int8_t *in0 = (int8_t*)info->dataptrs[0];
    uint16_t *out = (uint16_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint16_t)(*in0);
         in0 = (int8_t*)((char*)in0 + info->inner_strides[0]);
         out = (uint16_t*)((char*)out + info->inner_strides[1]);
    }
}

void i8_to_u32_cast(kernel_info *info) {
    int8_t *in0 = (int8_t*)info->dataptrs[0];
    uint32_t *out = (uint32_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint32_t)(*in0);
         in0 = (int8_t*)((char*)in0 + info->inner_strides[0]);
         out = (uint32_t*)((char*)out + info->inner_strides[1]);
    }
}

void i8_to_u64_cast(kernel_info *info) {
    int8_t *in0 = (int8_t*)info->dataptrs[0];
    uint64_t *out = (uint64_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint64_t)(*in0);
         in0 = (int8_t*)((char*)in0 + info->inner_strides[0]);
         out = (uint64_t*)((char*)out + info->inner_strides[1]);
    }
}

void i8_to_f32_cast(kernel_info *info) {
    int8_t *in0 = (int8_t*)info->dataptrs[0];
    float *out = (float*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (float)(*in0);
         in0 = (int8_t*)((char*)in0 + info->inner_strides[0]);
         out = (float*)((char*)out + info->inner_strides[1]);
    }
}

void i8_to_f64_cast(kernel_info *info) {
    int8_t *in0 = (int8_t*)info->dataptrs[0];
    double *out = (double*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (double)(*in0);
         in0 = (int8_t*)((char*)in0 + info->inner_strides[0]);
         out = (double*)((char*)out + info->inner_strides[1]);
    }
}

void i8_to_bool_cast(kernel_info *info) {
    int8_t *in0 = (int8_t*)info->dataptrs[0];
    bool *out = (bool*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (bool)(*in0);
         in0 = (int8_t*)((char*)in0 + info->inner_strides[0]);
         out = (bool*)((char*)out + info->inner_strides[1]);
    }
}

void i16_to_i8_cast(kernel_info *info) {
    int16_t *in0 = (int16_t*)info->dataptrs[0];
    int8_t *out = (int8_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int8_t)(*in0);
         in0 = (int16_t*)((char*)in0 + info->inner_strides[0]);
         out = (int8_t*)((char*)out + info->inner_strides[1]);
    }
}

void i16_to_i32_cast(kernel_info *info) {
    int16_t *in0 = (int16_t*)info->dataptrs[0];
    int32_t *out = (int32_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int32_t)(*in0);
         in0 = (int16_t*)((char*)in0 + info->inner_strides[0]);
         out = (int32_t*)((char*)out + info->inner_strides[1]);
    }
}

void i16_to_i64_cast(kernel_info *info) {
    int16_t *in0 = (int16_t*)info->dataptrs[0];
    int64_t *out = (int64_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int64_t)(*in0);
         in0 = (int16_t*)((char*)in0 + info->inner_strides[0]);
         out = (int64_t*)((char*)out + info->inner_strides[1]);
    }
}

void i16_to_u8_cast(kernel_info *info) {
    int16_t *in0 = (int16_t*)info->dataptrs[0];
    uint8_t *out = (uint8_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint8_t)(*in0);
         in0 = (int16_t*)((char*)in0 + info->inner_strides[0]);
         out = (uint8_t*)((char*)out + info->inner_strides[1]);
    }
}

void i16_to_u16_cast(kernel_info *info) {
    int16_t *in0 = (int16_t*)info->dataptrs[0];
    uint16_t *out = (uint16_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint16_t)(*in0);
         in0 = (int16_t*)((char*)in0 + info->inner_strides[0]);
         out = (uint16_t*)((char*)out + info->inner_strides[1]);
    }
}

void i16_to_u32_cast(kernel_info *info) {
    int16_t *in0 = (int16_t*)info->dataptrs[0];
    uint32_t *out = (uint32_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint32_t)(*in0);
         in0 = (int16_t*)((char*)in0 + info->inner_strides[0]);
         out = (uint32_t*)((char*)out + info->inner_strides[1]);
    }
}

void i16_to_u64_cast(kernel_info *info) {
    int16_t *in0 = (int16_t*)info->dataptrs[0];
    uint64_t *out = (uint64_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint64_t)(*in0);
         in0 = (int16_t*)((char*)in0 + info->inner_strides[0]);
         out = (uint64_t*)((char*)out + info->inner_strides[1]);
    }
}

void i16_to_f32_cast(kernel_info *info) {
    int16_t *in0 = (int16_t*)info->dataptrs[0];
    float *out = (float*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (float)(*in0);
         in0 = (int16_t*)((char*)in0 + info->inner_strides[0]);
         out = (float*)((char*)out + info->inner_strides[1]);
    }
}

void i16_to_f64_cast(kernel_info *info) {
    int16_t *in0 = (int16_t*)info->dataptrs[0];
    double *out = (double*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (double)(*in0);
         in0 = (int16_t*)((char*)in0 + info->inner_strides[0]);
         out = (double*)((char*)out + info->inner_strides[1]);
    }
}

void i16_to_bool_cast(kernel_info *info) {
    int16_t *in0 = (int16_t*)info->dataptrs[0];
    bool *out = (bool*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (bool)(*in0);
         in0 = (int16_t*)((char*)in0 + info->inner_strides[0]);
         out = (bool*)((char*)out + info->inner_strides[1]);
    }
}

void i32_to_i8_cast(kernel_info *info) {
    int32_t *in0 = (int32_t*)info->dataptrs[0];
    int8_t *out = (int8_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int8_t)(*in0);
         in0 = (int32_t*)((char*)in0 + info->inner_strides[0]);
         out = (int8_t*)((char*)out + info->inner_strides[1]);
    }
}

void i32_to_i16_cast(kernel_info *info) {
    int32_t *in0 = (int32_t*)info->dataptrs[0];
    int16_t *out = (int16_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int16_t)(*in0);
         in0 = (int32_t*)((char*)in0 + info->inner_strides[0]);
         out = (int16_t*)((char*)out + info->inner_strides[1]);
    }
}

void i32_to_i64_cast(kernel_info *info) {
    int32_t *in0 = (int32_t*)info->dataptrs[0];
    int64_t *out = (int64_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int64_t)(*in0);
         in0 = (int32_t*)((char*)in0 + info->inner_strides[0]);
         out = (int64_t*)((char*)out + info->inner_strides[1]);
    }
}

void i32_to_u8_cast(kernel_info *info) {
    int32_t *in0 = (int32_t*)info->dataptrs[0];
    uint8_t *out = (uint8_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint8_t)(*in0);
         in0 = (int32_t*)((char*)in0 + info->inner_strides[0]);
         out = (uint8_t*)((char*)out + info->inner_strides[1]);
    }
}

void i32_to_u16_cast(kernel_info *info) {
    int32_t *in0 = (int32_t*)info->dataptrs[0];
    uint16_t *out = (uint16_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint16_t)(*in0);
         in0 = (int32_t*)((char*)in0 + info->inner_strides[0]);
         out = (uint16_t*)((char*)out + info->inner_strides[1]);
    }
}

void i32_to_u32_cast(kernel_info *info) {
    int32_t *in0 = (int32_t*)info->dataptrs[0];
    uint32_t *out = (uint32_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint32_t)(*in0);
         in0 = (int32_t*)((char*)in0 + info->inner_strides[0]);
         out = (uint32_t*)((char*)out + info->inner_strides[1]);
    }
}

void i32_to_u64_cast(kernel_info *info) {
    int32_t *in0 = (int32_t*)info->dataptrs[0];
    uint64_t *out = (uint64_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint64_t)(*in0);
         in0 = (int32_t*)((char*)in0 + info->inner_strides[0]);
         out = (uint64_t*)((char*)out + info->inner_strides[1]);
    }
}

void i32_to_f32_cast(kernel_info *info) {
    int32_t *in0 = (int32_t*)info->dataptrs[0];
    float *out = (float*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (float)(*in0);
         in0 = (int32_t*)((char*)in0 + info->inner_strides[0]);
         out = (float*)((char*)out + info->inner_strides[1]);
    }
}

void i32_to_f64_cast(kernel_info *info) {
    int32_t *in0 = (int32_t*)info->dataptrs[0];
    double *out = (double*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (double)(*in0);
         in0 = (int32_t*)((char*)in0 + info->inner_strides[0]);
         out = (double*)((char*)out + info->inner_strides[1]);
    }
}

void i32_to_bool_cast(kernel_info *info) {
    int32_t *in0 = (int32_t*)info->dataptrs[0];
    bool *out = (bool*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (bool)(*in0);
         in0 = (int32_t*)((char*)in0 + info->inner_strides[0]);
         out = (bool*)((char*)out + info->inner_strides[1]);
    }
}

void i64_to_i8_cast(kernel_info *info) {
    int64_t *in0 = (int64_t*)info->dataptrs[0];
    int8_t *out = (int8_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int8_t)(*in0);
         in0 = (int64_t*)((char*)in0 + info->inner_strides[0]);
         out = (int8_t*)((char*)out + info->inner_strides[1]);
    }
}

void i64_to_i16_cast(kernel_info *info) {
    int64_t *in0 = (int64_t*)info->dataptrs[0];
    int16_t *out = (int16_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int16_t)(*in0);
         in0 = (int64_t*)((char*)in0 + info->inner_strides[0]);
         out = (int16_t*)((char*)out + info->inner_strides[1]);
    }
}

void i64_to_i32_cast(kernel_info *info) {
    int64_t *in0 = (int64_t*)info->dataptrs[0];
    int32_t *out = (int32_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int32_t)(*in0);
         in0 = (int64_t*)((char*)in0 + info->inner_strides[0]);
         out = (int32_t*)((char*)out + info->inner_strides[1]);
    }
}

void i64_to_u8_cast(kernel_info *info) {
    int64_t *in0 = (int64_t*)info->dataptrs[0];
    uint8_t *out = (uint8_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint8_t)(*in0);
         in0 = (int64_t*)((char*)in0 + info->inner_strides[0]);
         out = (uint8_t*)((char*)out + info->inner_strides[1]);
    }
}

void i64_to_u16_cast(kernel_info *info) {
    int64_t *in0 = (int64_t*)info->dataptrs[0];
    uint16_t *out = (uint16_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint16_t)(*in0);
         in0 = (int64_t*)((char*)in0 + info->inner_strides[0]);
         out = (uint16_t*)((char*)out + info->inner_strides[1]);
    }
}

void i64_to_u32_cast(kernel_info *info) {
    int64_t *in0 = (int64_t*)info->dataptrs[0];
    uint32_t *out = (uint32_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint32_t)(*in0);
         in0 = (int64_t*)((char*)in0 + info->inner_strides[0]);
         out = (uint32_t*)((char*)out + info->inner_strides[1]);
    }
}

void i64_to_u64_cast(kernel_info *info) {
    int64_t *in0 = (int64_t*)info->dataptrs[0];
    uint64_t *out = (uint64_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint64_t)(*in0);
         in0 = (int64_t*)((char*)in0 + info->inner_strides[0]);
         out = (uint64_t*)((char*)out + info->inner_strides[1]);
    }
}

void i64_to_f32_cast(kernel_info *info) {
    int64_t *in0 = (int64_t*)info->dataptrs[0];
    float *out = (float*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (float)(*in0);
         in0 = (int64_t*)((char*)in0 + info->inner_strides[0]);
         out = (float*)((char*)out + info->inner_strides[1]);
    }
}

void i64_to_f64_cast(kernel_info *info) {
    int64_t *in0 = (int64_t*)info->dataptrs[0];
    double *out = (double*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (double)(*in0);
         in0 = (int64_t*)((char*)in0 + info->inner_strides[0]);
         out = (double*)((char*)out + info->inner_strides[1]);
    }
}

void i64_to_bool_cast(kernel_info *info) {
    int64_t *in0 = (int64_t*)info->dataptrs[0];
    bool *out = (bool*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (bool)(*in0);
         in0 = (int64_t*)((char*)in0 + info->inner_strides[0]);
         out = (bool*)((char*)out + info->inner_strides[1]);
    }
}

void u8_to_i8_cast(kernel_info *info) {
    uint8_t *in0 = (uint8_t*)info->dataptrs[0];
    int8_t *out = (int8_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int8_t)(*in0);
         in0 = (uint8_t*)((char*)in0 + info->inner_strides[0]);
         out = (int8_t*)((char*)out + info->inner_strides[1]);
    }
}

void u8_to_i16_cast(kernel_info *info) {
    uint8_t *in0 = (uint8_t*)info->dataptrs[0];
    int16_t *out = (int16_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int16_t)(*in0);
         in0 = (uint8_t*)((char*)in0 + info->inner_strides[0]);
         out = (int16_t*)((char*)out + info->inner_strides[1]);
    }
}

void u8_to_i32_cast(kernel_info *info) {
    uint8_t *in0 = (uint8_t*)info->dataptrs[0];
    int32_t *out = (int32_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int32_t)(*in0);
         in0 = (uint8_t*)((char*)in0 + info->inner_strides[0]);
         out = (int32_t*)((char*)out + info->inner_strides[1]);
    }
}

void u8_to_i64_cast(kernel_info *info) {
    uint8_t *in0 = (uint8_t*)info->dataptrs[0];
    int64_t *out = (int64_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int64_t)(*in0);
         in0 = (uint8_t*)((char*)in0 + info->inner_strides[0]);
         out = (int64_t*)((char*)out + info->inner_strides[1]);
    }
}

void u8_to_u16_cast(kernel_info *info) {
    uint8_t *in0 = (uint8_t*)info->dataptrs[0];
    uint16_t *out = (uint16_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint16_t)(*in0);
         in0 = (uint8_t*)((char*)in0 + info->inner_strides[0]);
         out = (uint16_t*)((char*)out + info->inner_strides[1]);
    }
}

void u8_to_u32_cast(kernel_info *info) {
    uint8_t *in0 = (uint8_t*)info->dataptrs[0];
    uint32_t *out = (uint32_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint32_t)(*in0);
         in0 = (uint8_t*)((char*)in0 + info->inner_strides[0]);
         out = (uint32_t*)((char*)out + info->inner_strides[1]);
    }
}

void u8_to_u64_cast(kernel_info *info) {
    uint8_t *in0 = (uint8_t*)info->dataptrs[0];
    uint64_t *out = (uint64_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint64_t)(*in0);
         in0 = (uint8_t*)((char*)in0 + info->inner_strides[0]);
         out = (uint64_t*)((char*)out + info->inner_strides[1]);
    }
}

void u8_to_f32_cast(kernel_info *info) {
    uint8_t *in0 = (uint8_t*)info->dataptrs[0];
    float *out = (float*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (float)(*in0);
         in0 = (uint8_t*)((char*)in0 + info->inner_strides[0]);
         out = (float*)((char*)out + info->inner_strides[1]);
    }
}

void u8_to_f64_cast(kernel_info *info) {
    uint8_t *in0 = (uint8_t*)info->dataptrs[0];
    double *out = (double*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (double)(*in0);
         in0 = (uint8_t*)((char*)in0 + info->inner_strides[0]);
         out = (double*)((char*)out + info->inner_strides[1]);
    }
}

void u8_to_bool_cast(kernel_info *info) {
    uint8_t *in0 = (uint8_t*)info->dataptrs[0];
    bool *out = (bool*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (bool)(*in0);
         in0 = (uint8_t*)((char*)in0 + info->inner_strides[0]);
         out = (bool*)((char*)out + info->inner_strides[1]);
    }
}

void u16_to_i8_cast(kernel_info *info) {
    uint16_t *in0 = (uint16_t*)info->dataptrs[0];
    int8_t *out = (int8_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int8_t)(*in0);
         in0 = (uint16_t*)((char*)in0 + info->inner_strides[0]);
         out = (int8_t*)((char*)out + info->inner_strides[1]);
    }
}

void u16_to_i16_cast(kernel_info *info) {
    uint16_t *in0 = (uint16_t*)info->dataptrs[0];
    int16_t *out = (int16_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int16_t)(*in0);
         in0 = (uint16_t*)((char*)in0 + info->inner_strides[0]);
         out = (int16_t*)((char*)out + info->inner_strides[1]);
    }
}

void u16_to_i32_cast(kernel_info *info) {
    uint16_t *in0 = (uint16_t*)info->dataptrs[0];
    int32_t *out = (int32_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int32_t)(*in0);
         in0 = (uint16_t*)((char*)in0 + info->inner_strides[0]);
         out = (int32_t*)((char*)out + info->inner_strides[1]);
    }
}

void u16_to_i64_cast(kernel_info *info) {
    uint16_t *in0 = (uint16_t*)info->dataptrs[0];
    int64_t *out = (int64_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int64_t)(*in0);
         in0 = (uint16_t*)((char*)in0 + info->inner_strides[0]);
         out = (int64_t*)((char*)out + info->inner_strides[1]);
    }
}

void u16_to_u8_cast(kernel_info *info) {
    uint16_t *in0 = (uint16_t*)info->dataptrs[0];
    uint8_t *out = (uint8_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint8_t)(*in0);
         in0 = (uint16_t*)((char*)in0 + info->inner_strides[0]);
         out = (uint8_t*)((char*)out + info->inner_strides[1]);
    }
}

void u16_to_u32_cast(kernel_info *info) {
    uint16_t *in0 = (uint16_t*)info->dataptrs[0];
    uint32_t *out = (uint32_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint32_t)(*in0);
         in0 = (uint16_t*)((char*)in0 + info->inner_strides[0]);
         out = (uint32_t*)((char*)out + info->inner_strides[1]);
    }
}

void u16_to_u64_cast(kernel_info *info) {
    uint16_t *in0 = (uint16_t*)info->dataptrs[0];
    uint64_t *out = (uint64_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint64_t)(*in0);
         in0 = (uint16_t*)((char*)in0 + info->inner_strides[0]);
         out = (uint64_t*)((char*)out + info->inner_strides[1]);
    }
}

void u16_to_f32_cast(kernel_info *info) {
    uint16_t *in0 = (uint16_t*)info->dataptrs[0];
    float *out = (float*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (float)(*in0);
         in0 = (uint16_t*)((char*)in0 + info->inner_strides[0]);
         out = (float*)((char*)out + info->inner_strides[1]);
    }
}

void u16_to_f64_cast(kernel_info *info) {
    uint16_t *in0 = (uint16_t*)info->dataptrs[0];
    double *out = (double*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (double)(*in0);
         in0 = (uint16_t*)((char*)in0 + info->inner_strides[0]);
         out = (double*)((char*)out + info->inner_strides[1]);
    }
}

void u16_to_bool_cast(kernel_info *info) {
    uint16_t *in0 = (uint16_t*)info->dataptrs[0];
    bool *out = (bool*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (bool)(*in0);
         in0 = (uint16_t*)((char*)in0 + info->inner_strides[0]);
         out = (bool*)((char*)out + info->inner_strides[1]);
    }
}

void u32_to_i8_cast(kernel_info *info) {
    uint32_t *in0 = (uint32_t*)info->dataptrs[0];
    int8_t *out = (int8_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int8_t)(*in0);
         in0 = (uint32_t*)((char*)in0 + info->inner_strides[0]);
         out = (int8_t*)((char*)out + info->inner_strides[1]);
    }
}

void u32_to_i16_cast(kernel_info *info) {
    uint32_t *in0 = (uint32_t*)info->dataptrs[0];
    int16_t *out = (int16_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int16_t)(*in0);
         in0 = (uint32_t*)((char*)in0 + info->inner_strides[0]);
         out = (int16_t*)((char*)out + info->inner_strides[1]);
    }
}

void u32_to_i32_cast(kernel_info *info) {
    uint32_t *in0 = (uint32_t*)info->dataptrs[0];
    int32_t *out = (int32_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int32_t)(*in0);
         in0 = (uint32_t*)((char*)in0 + info->inner_strides[0]);
         out = (int32_t*)((char*)out + info->inner_strides[1]);
    }
}

void u32_to_i64_cast(kernel_info *info) {
    uint32_t *in0 = (uint32_t*)info->dataptrs[0];
    int64_t *out = (int64_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int64_t)(*in0);
         in0 = (uint32_t*)((char*)in0 + info->inner_strides[0]);
         out = (int64_t*)((char*)out + info->inner_strides[1]);
    }
}

void u32_to_u8_cast(kernel_info *info) {
    uint32_t *in0 = (uint32_t*)info->dataptrs[0];
    uint8_t *out = (uint8_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint8_t)(*in0);
         in0 = (uint32_t*)((char*)in0 + info->inner_strides[0]);
         out = (uint8_t*)((char*)out + info->inner_strides[1]);
    }
}

void u32_to_u16_cast(kernel_info *info) {
    uint32_t *in0 = (uint32_t*)info->dataptrs[0];
    uint16_t *out = (uint16_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint16_t)(*in0);
         in0 = (uint32_t*)((char*)in0 + info->inner_strides[0]);
         out = (uint16_t*)((char*)out + info->inner_strides[1]);
    }
}

void u32_to_u64_cast(kernel_info *info) {
    uint32_t *in0 = (uint32_t*)info->dataptrs[0];
    uint64_t *out = (uint64_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint64_t)(*in0);
         in0 = (uint32_t*)((char*)in0 + info->inner_strides[0]);
         out = (uint64_t*)((char*)out + info->inner_strides[1]);
    }
}

void u32_to_f32_cast(kernel_info *info) {
    uint32_t *in0 = (uint32_t*)info->dataptrs[0];
    float *out = (float*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (float)(*in0);
         in0 = (uint32_t*)((char*)in0 + info->inner_strides[0]);
         out = (float*)((char*)out + info->inner_strides[1]);
    }
}

void u32_to_f64_cast(kernel_info *info) {
    uint32_t *in0 = (uint32_t*)info->dataptrs[0];
    double *out = (double*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (double)(*in0);
         in0 = (uint32_t*)((char*)in0 + info->inner_strides[0]);
         out = (double*)((char*)out + info->inner_strides[1]);
    }
}

void u32_to_bool_cast(kernel_info *info) {
    uint32_t *in0 = (uint32_t*)info->dataptrs[0];
    bool *out = (bool*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (bool)(*in0);
         in0 = (uint32_t*)((char*)in0 + info->inner_strides[0]);
         out = (bool*)((char*)out + info->inner_strides[1]);
    }
}

void u64_to_i8_cast(kernel_info *info) {
    uint64_t *in0 = (uint64_t*)info->dataptrs[0];
    int8_t *out = (int8_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int8_t)(*in0);
         in0 = (uint64_t*)((char*)in0 + info->inner_strides[0]);
         out = (int8_t*)((char*)out + info->inner_strides[1]);
    }
}

void u64_to_i16_cast(kernel_info *info) {
    uint64_t *in0 = (uint64_t*)info->dataptrs[0];
    int16_t *out = (int16_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int16_t)(*in0);
         in0 = (uint64_t*)((char*)in0 + info->inner_strides[0]);
         out = (int16_t*)((char*)out + info->inner_strides[1]);
    }
}

void u64_to_i32_cast(kernel_info *info) {
    uint64_t *in0 = (uint64_t*)info->dataptrs[0];
    int32_t *out = (int32_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int32_t)(*in0);
         in0 = (uint64_t*)((char*)in0 + info->inner_strides[0]);
         out = (int32_t*)((char*)out + info->inner_strides[1]);
    }
}

void u64_to_i64_cast(kernel_info *info) {
    uint64_t *in0 = (uint64_t*)info->dataptrs[0];
    int64_t *out = (int64_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int64_t)(*in0);
         in0 = (uint64_t*)((char*)in0 + info->inner_strides[0]);
         out = (int64_t*)((char*)out + info->inner_strides[1]);
    }
}

void u64_to_u8_cast(kernel_info *info) {
    uint64_t *in0 = (uint64_t*)info->dataptrs[0];
    uint8_t *out = (uint8_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint8_t)(*in0);
         in0 = (uint64_t*)((char*)in0 + info->inner_strides[0]);
         out = (uint8_t*)((char*)out + info->inner_strides[1]);
    }
}

void u64_to_u16_cast(kernel_info *info) {
    uint64_t *in0 = (uint64_t*)info->dataptrs[0];
    uint16_t *out = (uint16_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint16_t)(*in0);
         in0 = (uint64_t*)((char*)in0 + info->inner_strides[0]);
         out = (uint16_t*)((char*)out + info->inner_strides[1]);
    }
}

void u64_to_u32_cast(kernel_info *info) {
    uint64_t *in0 = (uint64_t*)info->dataptrs[0];
    uint32_t *out = (uint32_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint32_t)(*in0);
         in0 = (uint64_t*)((char*)in0 + info->inner_strides[0]);
         out = (uint32_t*)((char*)out + info->inner_strides[1]);
    }
}

void u64_to_f32_cast(kernel_info *info) {
    uint64_t *in0 = (uint64_t*)info->dataptrs[0];
    float *out = (float*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (float)(*in0);
         in0 = (uint64_t*)((char*)in0 + info->inner_strides[0]);
         out = (float*)((char*)out + info->inner_strides[1]);
    }
}

void u64_to_f64_cast(kernel_info *info) {
    uint64_t *in0 = (uint64_t*)info->dataptrs[0];
    double *out = (double*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (double)(*in0);
         in0 = (uint64_t*)((char*)in0 + info->inner_strides[0]);
         out = (double*)((char*)out + info->inner_strides[1]);
    }
}

void u64_to_bool_cast(kernel_info *info) {
    uint64_t *in0 = (uint64_t*)info->dataptrs[0];
    bool *out = (bool*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (bool)(*in0);
         in0 = (uint64_t*)((char*)in0 + info->inner_strides[0]);
         out = (bool*)((char*)out + info->inner_strides[1]);
    }
}

void f32_to_i8_cast(kernel_info *info) {
    float *in0 = (float*)info->dataptrs[0];
    int8_t *out = (int8_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int8_t)(*in0);
         in0 = (float*)((char*)in0 + info->inner_strides[0]);
         out = (int8_t*)((char*)out + info->inner_strides[1]);
    }
}

void f32_to_i16_cast(kernel_info *info) {
    float *in0 = (float*)info->dataptrs[0];
    int16_t *out = (int16_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int16_t)(*in0);
         in0 = (float*)((char*)in0 + info->inner_strides[0]);
         out = (int16_t*)((char*)out + info->inner_strides[1]);
    }
}

void f32_to_i32_cast(kernel_info *info) {
    float *in0 = (float*)info->dataptrs[0];
    int32_t *out = (int32_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int32_t)(*in0);
         in0 = (float*)((char*)in0 + info->inner_strides[0]);
         out = (int32_t*)((char*)out + info->inner_strides[1]);
    }
}

void f32_to_i64_cast(kernel_info *info) {
    float *in0 = (float*)info->dataptrs[0];
    int64_t *out = (int64_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int64_t)(*in0);
         in0 = (float*)((char*)in0 + info->inner_strides[0]);
         out = (int64_t*)((char*)out + info->inner_strides[1]);
    }
}

void f32_to_u8_cast(kernel_info *info) {
    float *in0 = (float*)info->dataptrs[0];
    uint8_t *out = (uint8_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint8_t)(*in0);
         in0 = (float*)((char*)in0 + info->inner_strides[0]);
         out = (uint8_t*)((char*)out + info->inner_strides[1]);
    }
}

void f32_to_u16_cast(kernel_info *info) {
    float *in0 = (float*)info->dataptrs[0];
    uint16_t *out = (uint16_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint16_t)(*in0);
         in0 = (float*)((char*)in0 + info->inner_strides[0]);
         out = (uint16_t*)((char*)out + info->inner_strides[1]);
    }
}

void f32_to_u32_cast(kernel_info *info) {
    float *in0 = (float*)info->dataptrs[0];
    uint32_t *out = (uint32_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint32_t)(*in0);
         in0 = (float*)((char*)in0 + info->inner_strides[0]);
         out = (uint32_t*)((char*)out + info->inner_strides[1]);
    }
}

void f32_to_u64_cast(kernel_info *info) {
    float *in0 = (float*)info->dataptrs[0];
    uint64_t *out = (uint64_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint64_t)(*in0);
         in0 = (float*)((char*)in0 + info->inner_strides[0]);
         out = (uint64_t*)((char*)out + info->inner_strides[1]);
    }
}

void f32_to_f64_cast(kernel_info *info) {
    float *in0 = (float*)info->dataptrs[0];
    double *out = (double*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (double)(*in0);
         in0 = (float*)((char*)in0 + info->inner_strides[0]);
         out = (double*)((char*)out + info->inner_strides[1]);
    }
}

void f32_to_bool_cast(kernel_info *info) {
    float *in0 = (float*)info->dataptrs[0];
    bool *out = (bool*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (bool)(*in0);
         in0 = (float*)((char*)in0 + info->inner_strides[0]);
         out = (bool*)((char*)out + info->inner_strides[1]);
    }
}

void f64_to_i8_cast(kernel_info *info) {
    double *in0 = (double*)info->dataptrs[0];
    int8_t *out = (int8_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int8_t)(*in0);
         in0 = (double*)((char*)in0 + info->inner_strides[0]);
         out = (int8_t*)((char*)out + info->inner_strides[1]);
    }
}

void f64_to_i16_cast(kernel_info *info) {
    double *in0 = (double*)info->dataptrs[0];
    int16_t *out = (int16_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int16_t)(*in0);
         in0 = (double*)((char*)in0 + info->inner_strides[0]);
         out = (int16_t*)((char*)out + info->inner_strides[1]);
    }
}

void f64_to_i32_cast(kernel_info *info) {
    double *in0 = (double*)info->dataptrs[0];
    int32_t *out = (int32_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int32_t)(*in0);
         in0 = (double*)((char*)in0 + info->inner_strides[0]);
         out = (int32_t*)((char*)out + info->inner_strides[1]);
    }
}

void f64_to_i64_cast(kernel_info *info) {
    double *in0 = (double*)info->dataptrs[0];
    int64_t *out = (int64_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int64_t)(*in0);
         in0 = (double*)((char*)in0 + info->inner_strides[0]);
         out = (int64_t*)((char*)out + info->inner_strides[1]);
    }
}

void f64_to_u8_cast(kernel_info *info) {
    double *in0 = (double*)info->dataptrs[0];
    uint8_t *out = (uint8_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint8_t)(*in0);
         in0 = (double*)((char*)in0 + info->inner_strides[0]);
         out = (uint8_t*)((char*)out + info->inner_strides[1]);
    }
}

void f64_to_u16_cast(kernel_info *info) {
    double *in0 = (double*)info->dataptrs[0];
    uint16_t *out = (uint16_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint16_t)(*in0);
         in0 = (double*)((char*)in0 + info->inner_strides[0]);
         out = (uint16_t*)((char*)out + info->inner_strides[1]);
    }
}

void f64_to_u32_cast(kernel_info *info) {
    double *in0 = (double*)info->dataptrs[0];
    uint32_t *out = (uint32_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint32_t)(*in0);
         in0 = (double*)((char*)in0 + info->inner_strides[0]);
         out = (uint32_t*)((char*)out + info->inner_strides[1]);
    }
}

void f64_to_u64_cast(kernel_info *info) {
    double *in0 = (double*)info->dataptrs[0];
    uint64_t *out = (uint64_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint64_t)(*in0);
         in0 = (double*)((char*)in0 + info->inner_strides[0]);
         out = (uint64_t*)((char*)out + info->inner_strides[1]);
    }
}

void f64_to_f32_cast(kernel_info *info) {
    double *in0 = (double*)info->dataptrs[0];
    float *out = (float*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (float)(*in0);
         in0 = (double*)((char*)in0 + info->inner_strides[0]);
         out = (float*)((char*)out + info->inner_strides[1]);
    }
}

void f64_to_bool_cast(kernel_info *info) {
    double *in0 = (double*)info->dataptrs[0];
    bool *out = (bool*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (bool)(*in0);
         in0 = (double*)((char*)in0 + info->inner_strides[0]);
         out = (bool*)((char*)out + info->inner_strides[1]);
    }
}

void bool_to_i8_cast(kernel_info *info) {
    bool *in0 = (bool*)info->dataptrs[0];
    int8_t *out = (int8_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int8_t)(*in0);
         in0 = (bool*)((char*)in0 + info->inner_strides[0]);
         out = (int8_t*)((char*)out + info->inner_strides[1]);
    }
}

void bool_to_i16_cast(kernel_info *info) {
    bool *in0 = (bool*)info->dataptrs[0];
    int16_t *out = (int16_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int16_t)(*in0);
         in0 = (bool*)((char*)in0 + info->inner_strides[0]);
         out = (int16_t*)((char*)out + info->inner_strides[1]);
    }
}

void bool_to_i32_cast(kernel_info *info) {
    bool *in0 = (bool*)info->dataptrs[0];
    int32_t *out = (int32_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int32_t)(*in0);
         in0 = (bool*)((char*)in0 + info->inner_strides[0]);
         out = (int32_t*)((char*)out + info->inner_strides[1]);
    }
}

void bool_to_i64_cast(kernel_info *info) {
    bool *in0 = (bool*)info->dataptrs[0];
    int64_t *out = (int64_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (int64_t)(*in0);
         in0 = (bool*)((char*)in0 + info->inner_strides[0]);
         out = (int64_t*)((char*)out + info->inner_strides[1]);
    }
}

void bool_to_u8_cast(kernel_info *info) {
    bool *in0 = (bool*)info->dataptrs[0];
    uint8_t *out = (uint8_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint8_t)(*in0);
         in0 = (bool*)((char*)in0 + info->inner_strides[0]);
         out = (uint8_t*)((char*)out + info->inner_strides[1]);
    }
}

void bool_to_u16_cast(kernel_info *info) {
    bool *in0 = (bool*)info->dataptrs[0];
    uint16_t *out = (uint16_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint16_t)(*in0);
         in0 = (bool*)((char*)in0 + info->inner_strides[0]);
         out = (uint16_t*)((char*)out + info->inner_strides[1]);
    }
}

void bool_to_u32_cast(kernel_info *info) {
    bool *in0 = (bool*)info->dataptrs[0];
    uint32_t *out = (uint32_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint32_t)(*in0);
         in0 = (bool*)((char*)in0 + info->inner_strides[0]);
         out = (uint32_t*)((char*)out + info->inner_strides[1]);
    }
}

void bool_to_u64_cast(kernel_info *info) {
    bool *in0 = (bool*)info->dataptrs[0];
    uint64_t *out = (uint64_t*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (uint64_t)(*in0);
         in0 = (bool*)((char*)in0 + info->inner_strides[0]);
         out = (uint64_t*)((char*)out + info->inner_strides[1]);
    }
}

void bool_to_f32_cast(kernel_info *info) {
    bool *in0 = (bool*)info->dataptrs[0];
    float *out = (float*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (float)(*in0);
         in0 = (bool*)((char*)in0 + info->inner_strides[0]);
         out = (float*)((char*)out + info->inner_strides[1]);
    }
}

void bool_to_f64_cast(kernel_info *info) {
    bool *in0 = (bool*)info->dataptrs[0];
    double *out = (double*)info->dataptrs[1];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {
        *out = (double)(*in0);
         in0 = (bool*)((char*)in0 + info->inner_strides[0]);
         out = (double*)((char*)out + info->inner_strides[1]);
    }
}

