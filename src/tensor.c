#include "tensor.h"

Tensor *tensor_create(uint64_t ndim, uint64_t *shape, Backend backend, dtype type, void *data) {
    Value *self;
    Tensor *tensor = malloc(sizeof(Tensor));

    if (!tensor) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "");
        return NULL;
    }

    if (value_create(&self, data, type, ndim, shape, backend) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create");
        free(tensor);
        return NULL;
    }

    tensor->self = self;
    return tensor;
}

Tensor *tensor_zeros(uint64_t ndim, uint64_t *shape, dtype type, Backend backend, bool requires_grad) { 
    Value *out_self;
    Tensor *out = malloc(sizeof(Tensor));

    if (!out) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "");
        return NULL;
    }

    if (value_zeros(&out_self, ndim, shape, type, backend, requires_grad) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_zeros");
        free(out);
        return NULL;
    }

    out->self = out_self;
    return out;
}

Tensor *tensor_zeros_like(Tensor *a, bool requires_grad) {   
    if (!a) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "");
        return NULL;
    }

    Value *out_self;
    Tensor *out = malloc(sizeof(Tensor));

    if (!out) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "tensor_zeros_like");
        return NULL;
    }

    if (value_zeros_like(&out_self, a->self, requires_grad) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_zeros_like");
        free(out);
        return NULL;
    }

    out->self = out_self;
    return out;
}

Tensor *tensor_ones(uint64_t ndim, uint64_t *shape, dtype type, Backend backend, bool requires_grad) {    
    Value *out_self;
    Tensor *out = malloc(sizeof(Tensor));

    if (!out) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "");
        return NULL;
    }

    if (value_ones(&out_self, ndim, shape, type, backend, requires_grad) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_ones");
        free(out);
        return NULL;
    }

    out->self = out_self;
    return out;
}

Tensor *tensor_ones_like(Tensor *a, bool requires_grad) {    
    if (!a) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "");
        return NULL;
    }

    Value *out_self;
    Tensor *out = malloc(sizeof(Tensor));

    if (!out) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "tensor_ones_like");
        return NULL;
    }

    if (value_ones_like(&out_self, a->self, requires_grad) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_ones_like");
        free(out);
        return NULL;
    }

    out->self = out_self;
    return out;
}

Tensor *tensor_fill(uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double fill_value, bool requires_grad) {    
    Value *out_self;
    Tensor *out = malloc(sizeof(Tensor));

    if (!out) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "");
        return NULL;
    }

    if (value_fill(&out_self, ndim, shape, type, backend, fill_value, requires_grad) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_fill");
        free(out);
        return NULL;
    }

    out->self = out_self;
    return out;
}

Tensor *tensor_arange(uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double start, double step, bool requires_grad) {   
    Value *out_self;
    Tensor *out = malloc(sizeof(Tensor));

    if (!out) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "");
        return NULL;
    }

    if (value_arange(&out_self, ndim, shape, type, backend, start, step, requires_grad) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_arange");
        free(out);
        return NULL;
    }

    out->self = out_self;
    return out;
}

Tensor *tensor_linspace(uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double start, double stop, bool requires_grad) {   
    Value *out_self;
    Tensor *out = malloc(sizeof(Tensor));

    if (!out) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "");
        return NULL;
    }

    if (value_linspace(&out_self, ndim, shape, type, backend, start, stop, requires_grad) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_linspace");
        free(out);
        return NULL;
    }

    out->self = out_self;
    return out;
}

Tensor *tensor_randu(uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double low, double high, bool requires_grad) {   
    Value *out_self;
    Tensor *out = malloc(sizeof(Tensor));

    if (!out) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "");
        return NULL;
    }

    if (value_randu(&out_self, ndim, shape, type, backend, low, high, requires_grad) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_randu");
        free(out);
        return NULL;
    }

    out->self = out_self;
    return out;
}

Tensor *tensor_randn(uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double mean, double stdev, bool requires_grad) {    
    Value *out_self;
    Tensor *out = malloc(sizeof(Tensor));

    if (!out) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "");
        return NULL;
    }

    if (value_randn(&out_self, ndim, shape, type, backend, mean, stdev, requires_grad) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_randn");
        free(out);
        return NULL;
    }

    out->self = out_self;
    return out;
}

Tensor *tensor_randint(uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double low, double high, bool requires_grad) {   
    Value *out_self;
    Tensor *out = malloc(sizeof(Tensor));

    if (!out) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "");
        return NULL;
    }

    if (value_randint(&out_self, ndim, shape, type, backend, low, high, requires_grad) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_randint");
        free(out);
        return NULL;
    }

    out->self = out_self;
    return out;
}


Tensor *tensor_add(Tensor *a, Tensor *b) {
    if (!a || !b) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "");
        return NULL;
    }

    Value *out_self;
    Tensor *out = malloc(sizeof(Tensor));

    if (!out) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "tensor_add");
        return NULL;
    }

    if (value_add(&out_self, a->self, b->self) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_add");
        free(out);
        return NULL;
    }

    out->self = out_self;
    return out;
}

Tensor *tensor_sub(Tensor *a, Tensor *b) {
    if (!a || !b) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "");
        return NULL;
    }

    Value *out_self;
    Tensor *out = malloc(sizeof(Tensor));

    if (!out) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "tensor_sub");
        return NULL;
    }

    if (value_subtract(&out_self, a->self, b->self) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_subtract");
        free(out);
        return NULL;
    }

    out->self = out_self;
    return out;
}

Tensor *tensor_mul(Tensor *a, Tensor *b) {  
    if (!a || !b) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "");
        return NULL;
    }

    Value *out_self;
    Tensor *out = malloc(sizeof(Tensor));

    if (!out) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "tensor_mul");
        return NULL;
    }

    if (value_multiply(&out_self, a->self, b->self) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_multiply");
        free(out);
        return NULL;
    }

    out->self = out_self;
    return out;
}

Tensor *tensor_div(Tensor *a, Tensor *b) {   
    if (!a || !b) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "");
        return NULL;
    }

    Value *out_self;
    Tensor *out = malloc(sizeof(Tensor));

    if (!out) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "tensor_div");
        return NULL;
    }

    if (value_divide(&out_self, a->self, b->self) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_divide");
        free(out);
        return NULL;
    }

    out->self = out_self;
    return out;
}

Tensor *tensor_floor_div(Tensor *a, Tensor *b) {    
    if (!a || !b) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "");
        return NULL;
    }

    Value *out_self;
    Tensor *out = malloc(sizeof(Tensor));

    if (!out) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "tensor_floor_div");
        return NULL;
    }

    if (value_floor_divide(&out_self, a->self, b->self) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_floor_divide");
        free(out);
        return NULL;
    }

    out->self = out_self;
    return out;
}

Tensor *tensor_pow(Tensor *a, Tensor *b) {   
    if (!a || !b) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "");
        return NULL;
    }

    Value *out_self;
    Tensor *out = malloc(sizeof(Tensor));

    if (!out) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "tensor_pow");
        return NULL;
    }

    if (value_power(&out_self, a->self, b->self) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_power");
        free(out);
        return NULL;
    }

    out->self = out_self;
    return out;
}


Tensor *tensor_eq(Tensor *a, Tensor *b) {   
    if (!a || !b) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "");
        return NULL;
    }

    Value *out_self;
    Tensor *out = malloc(sizeof(Tensor));

    if (!out) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "tensor_eq");
        return NULL;
    }

    if (value_equality(&out_self, a->self, b->self) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_equality");
        free(out);
        return NULL;
    }

    out->self = out_self;
    return out;
}

Tensor *tensor_gt(Tensor *a, Tensor *b) {  
    if (!a || !b) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "");
        return NULL;
    }

    Value *out_self;
    Tensor *out = malloc(sizeof(Tensor));

    if (!out) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "tensor_gt");
        return NULL;
    }

    if (value_greater_than(&out_self, a->self, b->self) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_greater_than");
        free(out);
        return NULL;
    }

    out->self = out_self;
    return out;
}

Tensor *tensor_lt(Tensor *a, Tensor *b) {  
    if (!a || !b) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "");
        return NULL;
    }

    Value *out_self;
    Tensor *out = malloc(sizeof(Tensor));

    if (!out) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "tensor_lt");
        return NULL;
    }

    if (value_less_than(&out_self, a->self, b->self) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_less_than");
        free(out);
        return NULL;
    }

    out->self = out_self;
    return out;
}

Tensor *tensor_maximum(Tensor *a, Tensor *b) {  
    if (!a || !b) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "");
        return NULL;
    }

    Value *out_self;
    Tensor *out = malloc(sizeof(Tensor));

    if (!out) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "tensor_maximum");
        return NULL;
    }

    if (value_maximum(&out_self, a->self, b->self) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_maximum");
        free(out);
        return NULL;
    }

    out->self = out_self;
    return out;
}

Tensor *tensor_minimum(Tensor *a, Tensor *b) {   
    if (!a || !b) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "");
        return NULL;
    }

    Value *out_self;
    Tensor *out = malloc(sizeof(Tensor));

    if (!out) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "tensor_minimum");
        return NULL;
    }

    if (value_minimum(&out_self, a->self, b->self) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_minimum");
        free(out);
        return NULL;
    }

    out->self = out_self;
    return out;
}


Tensor *tensor_sqrt(Tensor *a) {   
    if (!a) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "");
        return NULL;
    }

    Value *out_self;
    Tensor *out = malloc(sizeof(Tensor));

    if (!out) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "tensor_sqrt");
        return NULL;
    }

    if (value_square_root(&out_self, a->self) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_square_root");
        free(out);
        return NULL;
    }

    out->self = out_self;
    return out;
}

Tensor *tensor_abs(Tensor *a) {  
    if (!a) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "");
        return NULL;
    }

    Value *out_self;
    Tensor *out = malloc(sizeof(Tensor));

    if (!out) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "tensor_abs");
        return NULL;
    }

    if (value_absolute_value(&out_self, a->self) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_absolute_value");
        free(out);
        return NULL;
    }

    out->self = out_self;
    return out;
}

Tensor *tensor_neg(Tensor *a) {  
    if (!a) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "");
        return NULL;
    }

    Value *out_self;
    Tensor *out = malloc(sizeof(Tensor));

    if (!out) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "tensor_neg");
        return NULL;
    }

    if (value_negation(&out_self, a->self) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_negation");
        free(out);
        return NULL;
    }

    out->self = out_self;
    return out;
}


Tensor *tensor_sum(Tensor *a, uint64_t rndim, uint64_t *dims, bool keepdim) {
    if (!a) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "");
        return NULL;
    }

    Value *out_self;
    Tensor *out = malloc(sizeof(Tensor));

    if (!out) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "tensor_sum");
        return NULL;
    }

    if (value_sum(&out_self, a->self, rndim, dims, keepdim) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_sum");
        free(out);
        return NULL;
    }

    out->self = out_self;
    return out;
}

Tensor *tensor_max(Tensor *a, uint64_t rndim, uint64_t *dims, bool keepdim) {  
    if (!a) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "");
        return NULL;
    }

    Value *out_self;
    Tensor *out = malloc(sizeof(Tensor));

    if (!out) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "tensor_max");
        return NULL;
    }

    if (value_max(&out_self, a->self, rndim, dims, keepdim) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_max");
        free(out);
        return NULL;
    }

    out->self = out_self;
    return out;
}

Tensor *tensor_min(Tensor *a, uint64_t rndim, uint64_t *dims, bool keepdim) {
    if (!a) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "");
        return NULL;
    }

    Value *out_self;
    Tensor *out = malloc(sizeof(Tensor));

    if (!out) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "tensor_min");
        return NULL;
    }

    if (value_min(&out_self, a->self, rndim, dims, keepdim) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_min");
        free(out);
        return NULL;
    }

    out->self = out_self;
    return out;
}


Tensor *tensor_matmul(Tensor *a, Tensor *b) {
    if (!a || !b) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "");
        return NULL;
    }

    Value *out_self;
    Tensor *out = malloc(sizeof(Tensor));

    if (!out) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "tensor_matmul");
        return NULL;
    }

    if (value_matmul(&out_self, a->self, b->self) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_matmul");
        free(out);
        return NULL;
    }

    out->self = out_self;
    return out;
}


bool grad(Tensor *a) {
    if (!a) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "");
        return false;
    }

    if (backward(a->self) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "backward");
        return false;
    }

    return true;
}

bool tensor_retain_grad(Tensor *a) { 
    if (!a) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "");
        return false;
    }

    if (value_retain_grad(a->self) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_retain_grad");
        return false;
    }

    return true;
}

void tensor_print(Tensor *a) {  
    if (!a) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "");
        return;
    }

    if (value_print_value(a->self, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_print_value");
    }
}

void tensor_print_grad(Tensor *a) {
    if (!a) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "");
        return;
    }

    if (value_print_value(a->self, true) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_print_value");
    }
}
