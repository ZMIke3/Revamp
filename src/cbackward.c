#include "Backward_kernels.h"


void f32_kernel_add_backward(kernel_info *info) {
    
	float *in0_grad = (float*)info->dataptrs[0];
	float *in1_grad = (float*)info->dataptrs[1];
	float *out_grad = (float*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {

        *in0_grad += *out_grad;
        *in1_grad += *out_grad;
    
        in0_grad = (float*)((char*)in0_grad + info->inner_strides[0]);
        in1_grad = (float*)((char*)in1_grad + info->inner_strides[1]);
        out_grad = (float*)((char*)out_grad + info->inner_strides[2]);
    }
}

void f64_kernel_add_backward(kernel_info *info) {
    
	double *in0_grad = (double*)info->dataptrs[0];
	double *in1_grad = (double*)info->dataptrs[1];
	double *out_grad = (double*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {

        *in0_grad += *out_grad;
        *in1_grad += *out_grad;
    
        in0_grad = (double*)((char*)in0_grad + info->inner_strides[0]);
        in1_grad = (double*)((char*)in1_grad + info->inner_strides[1]);
        out_grad = (double*)((char*)out_grad + info->inner_strides[2]);
    }
}


void f32_kernel_subtract_backward(kernel_info *info) {
    
	float *in0_grad = (float*)info->dataptrs[0];
	float *in1_grad = (float*)info->dataptrs[1];
	float *out_grad = (float*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {

        *in0_grad += *out_grad;
        *in1_grad -= *out_grad;
    
        in0_grad = (float*)((char*)in0_grad + info->inner_strides[0]);
        in1_grad = (float*)((char*)in1_grad + info->inner_strides[1]);
        out_grad = (float*)((char*)out_grad + info->inner_strides[2]);
    }
}

void f64_kernel_subtract_backward(kernel_info *info) {
    
	double *in0_grad = (double*)info->dataptrs[0];
	double *in1_grad = (double*)info->dataptrs[1];
	double *out_grad = (double*)info->dataptrs[2];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {

        *in0_grad += *out_grad;
        *in1_grad -= *out_grad;
    
        in0_grad = (double*)((char*)in0_grad + info->inner_strides[0]);
        in1_grad = (double*)((char*)in1_grad + info->inner_strides[1]);
        out_grad = (double*)((char*)out_grad + info->inner_strides[2]);
    }
}


void f32_kernel_multiply_backward(kernel_info *info) {
    
	float *in0_grad = (float*)info->dataptrs[0];
	float *in1_grad = (float*)info->dataptrs[1];
	float *in0_val = (float*)info->dataptrs[2];
	float *in1_val = (float*)info->dataptrs[3];
	float *out_val = (float*)info->dataptrs[4];
	float *out_grad = (float*)info->dataptrs[5];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {

        *in0_grad += *out_grad * *in1_val;
        *in1_grad += *out_grad * *in0_val;
    
        in0_grad = (float*)((char*)in0_grad + info->inner_strides[0]);
        in1_grad = (float*)((char*)in1_grad + info->inner_strides[1]);
        in0_val = (float*)((char*)in0_val + info->inner_strides[2]);
        in1_val = (float*)((char*)in1_val + info->inner_strides[3]);
        out_val = (float*)((char*)out_val + info->inner_strides[4]);
        out_grad = (float*)((char*)out_grad + info->inner_strides[5]);
    }
}

void f64_kernel_multiply_backward(kernel_info *info) {
    
	double *in0_grad = (double*)info->dataptrs[0];
	double *in1_grad = (double*)info->dataptrs[1];
	double *in0_val = (double*)info->dataptrs[2];
	double *in1_val = (double*)info->dataptrs[3];
	double *out_val = (double*)info->dataptrs[4];
	double *out_grad = (double*)info->dataptrs[5];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {

        *in0_grad += *out_grad * *in1_val;
        *in1_grad += *out_grad * *in0_val;
    
        in0_grad = (double*)((char*)in0_grad + info->inner_strides[0]);
        in1_grad = (double*)((char*)in1_grad + info->inner_strides[1]);
        in0_val = (double*)((char*)in0_val + info->inner_strides[2]);
        in1_val = (double*)((char*)in1_val + info->inner_strides[3]);
        out_val = (double*)((char*)out_val + info->inner_strides[4]);
        out_grad = (double*)((char*)out_grad + info->inner_strides[5]);
    }
}


void f32_kernel_divide_backward(kernel_info *info) {
    
	float *in0_grad = (float*)info->dataptrs[0];
	float *in1_grad = (float*)info->dataptrs[1];
	float *in0_val = (float*)info->dataptrs[2];
	float *in1_val = (float*)info->dataptrs[3];
	float *out_val = (float*)info->dataptrs[4];
	float *out_grad = (float*)info->dataptrs[5];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {

        *in0_grad += *out_grad * (1.0 / *in1_val);
        *in1_grad += *out_grad * (-(*in0_val) / (*in1_val * *in1_val));
    
        in0_grad = (float*)((char*)in0_grad + info->inner_strides[0]);
        in1_grad = (float*)((char*)in1_grad + info->inner_strides[1]);
        in0_val = (float*)((char*)in0_val + info->inner_strides[2]);
        in1_val = (float*)((char*)in1_val + info->inner_strides[3]);
        out_val = (float*)((char*)out_val + info->inner_strides[4]);
        out_grad = (float*)((char*)out_grad + info->inner_strides[5]);
    }
}

void f64_kernel_divide_backward(kernel_info *info) {
    
	double *in0_grad = (double*)info->dataptrs[0];
	double *in1_grad = (double*)info->dataptrs[1];
	double *in0_val = (double*)info->dataptrs[2];
	double *in1_val = (double*)info->dataptrs[3];
	double *out_val = (double*)info->dataptrs[4];
	double *out_grad = (double*)info->dataptrs[5];

    for (int64_t i = 0; i < info->inner_loop_size; i++) {

        *in0_grad += *out_grad * (1.0 / *in1_val);
        *in1_grad += *out_grad * (-(*in0_val) / (*in1_val * *in1_val));
    
        in0_grad = (double*)((char*)in0_grad + info->inner_strides[0]);
        in1_grad = (double*)((char*)in1_grad + info->inner_strides[1]);
        in0_val = (double*)((char*)in0_val + info->inner_strides[2]);
        in1_val = (double*)((char*)in1_val + info->inner_strides[3]);
        out_val = (double*)((char*)out_val + info->inner_strides[4]);
        out_grad = (double*)((char*)out_grad + info->inner_strides[5]);
    }
}

