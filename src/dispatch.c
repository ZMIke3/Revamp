#include "Dispatch.h"

kernel_fn kernel_table[DTYPE_COUNT * BACKEND_COUNT * KERNEL_COUNT];

int64_t kernel_neighborhood[3] = {(int64_t)DTYPE_COUNT, (int64_t)BACKEND_COUNT, (int64_t)KERNEL_COUNT};

bool kernel_init = false;

bool dispatch_register_kernel(dtype type, Backend backend, kernel_id id,  kernel_fn fn) {
    if (type >= DTYPE_COUNT || backend >= BACKEND_COUNT || id >= KERNEL_COUNT) {
        REPORT_ERROR(ERR_INPUT_OUT_OF_BOUNDS, "input must be: type <= %d && backend <= %d && id <= %d", DTYPE_COUNT, BACKEND_COUNT, KERNEL_COUNT);
        return false;
    }

    int64_t *stride;
    
    if (helper_calc_stride(&stride, 3, kernel_neighborhood, sizeof(kernel_fn)) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: helper_calc_stride");
        return false;
    }
   
    int64_t idx_bytes = stride[0] * (int64_t)type + stride[1] * (int64_t)backend + stride[2] * (int64_t)id;

    kernel_fn *target_ptr = (kernel_fn *)(((char *)kernel_table) + idx_bytes);

    *target_ptr = fn;

    free(stride);

    return true;
}

bool dispatch_get_kernel(kernel_fn *out_fn, dtype type, Backend backend, kernel_id id) {
    if (type >= DTYPE_COUNT || backend >= BACKEND_COUNT || id >= KERNEL_COUNT) {
        REPORT_ERROR(ERR_INPUT_OUT_OF_BOUNDS, "input must be: type <= %d && backend <= %d && id <= %d", DTYPE_COUNT, BACKEND_COUNT, KERNEL_COUNT);
        return false;
    }

    if (kernel_init != true) {
        register_all_kernels();
        kernel_init = true; 
    }
    
    int64_t *stride;
    
    if (helper_calc_stride(&stride, 3, kernel_neighborhood, sizeof(kernel_fn)) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "helper_calc_stride");
        return false;
    }

    int64_t idx_bytes = stride[0] * (int64_t)type + stride[1] * (int64_t)backend + stride[2] * (int64_t)id;

    kernel_fn target_ptr = *(kernel_fn *)(((char *)kernel_table) + idx_bytes);

    
    *out_fn = target_ptr;
    
    free(stride);

    return true;
}


