#include "vfuncs.h"

bool value_zeros(Value **out_value, uint64_t ndim, uint64_t *shape, dtype type, Backend backend, bool requires_grad) {
    
    if (!out_value) return false;

    array *out;

    if (array_zeros(&out, ndim, shape, type, backend) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_zeros");
        return false;
    }

    if (value_create_value(out_value, out, requires_grad, true) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
        return false;
    }

    return true;
}

bool value_zeros_like(Value **out_value, Value *in, bool requires_grad) {
    
    if (!out_value || !in || !in->data) return false;

    array *out;

    if (array_zeros_like(in->data, &out) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_zeros_like");
        return false;
    }

    if (value_create_value(out_value, out, requires_grad, true) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
        return false;
    }

    return true;
}

bool value_ones(Value **out_value, uint64_t ndim, uint64_t *shape, dtype type, Backend backend, bool requires_grad) {
    
    if (!out_value) return false;

    array *out;

    if (array_ones(&out, ndim, shape, type, backend) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_ones");
        return false;
    }

    if (value_create_value(out_value, out, requires_grad, true) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
        return false;
    }

    return true;
}

bool value_ones_like(Value **out_value, Value *in, bool requires_grad) {
    
    if (!out_value || !in || !in->data) return false;

    array *out;
    
    if (array_ones_like(in->data, &out) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_ones_like");
        return false;
    }

    if (value_create_value(out_value, out, requires_grad, true) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
        return false;
    }

    return true;
}

bool value_fill(Value **out_value, uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double fill_value, bool requires_grad) {
    
    if (!out_value) return false;

    array *out;
    
    if (array_fill(&out, ndim, shape, type, backend, fill_value) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_fill");
        return false;
    }

    if (value_create_value(out_value, out, requires_grad, true) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
        return false;
    }

    return true;
}

bool value_arange(Value **out_value, uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double start, double step, bool requires_grad) {
   
    if (!out_value) return false;

    array *out;
    if (array_arange(&out, ndim, shape, type, backend, start, step) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_arange");
        return false;
    }

    if (value_create_value(out_value, out, requires_grad, true) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
        return false;
    }

    return true;
}

bool value_linspace(Value **out_value, uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double start, double stop, bool requires_grad) {
    
    if (!out_value) return false;

    array *out;

    if (array_linspace(&out, ndim, shape, type, backend, start, stop) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_linspace");
        return false;
    }

    if (value_create_value(out_value, out, requires_grad, true) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
        return false;
    }

    return true;
}

bool value_randu(Value **out_value, uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double low, double high, bool requires_grad) {
   
    if (!out_value) return false;

    array *out;

    if (array_randu(&out, ndim, shape, type, backend, low, high) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_randu");
        return false;
    }

    if (value_create_value(out_value, out, requires_grad, true) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
        return false;
    }

    return true;
}

bool value_randn(Value **out_value, uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double mean, double stdev, bool requires_grad) {
    
    if (!out_value) return false;

    array *out;

    if (array_randn(&out, ndim, shape, type, backend, mean, stdev) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_randn");
        return false;
    }

    if (value_create_value(out_value, out, requires_grad, true) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
        return false;
    }

    return true;
}

bool value_randint(Value **out_value, uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double low, double high, bool requires_grad) {
    
    if (!out_value) return false;

    array *out;

    if (array_randint(&out, ndim, shape, type, backend, low, high) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_randint");
        return false;
    }

    if (value_create_value(out_value, out, requires_grad, true) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
        return false;
    }

    return true;
}



bool value_add(Value **out_value, Value *a, Value *b) {
    
    if (!out_value || !a || !b) return false;

    array *output;
    
    if (array_add(&output, a->data, b->data) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_add");
        return false;
    }

    bool req_grad = a->requires_grad || b->requires_grad;
    if (value_create_value(out_value, output, req_grad, !req_grad) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
        return false;
    }

    if (req_grad) {
        Function *fn;
        if (function_create_function(&fn, array_add_backward) == false) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_create_function");
            return false;
        }
        if (!function_append_input(fn, a)) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_append_input");
            return false;
        }
        if (!function_append_input(fn, b)) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_append_input");
            return false;
        }
        if (value_set_creator(*out_value, fn) == false) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_set_creator");
            return false;
        }
    }

    return true;
}

bool value_subtract(Value **out_value, Value *a, Value *b) {
    if (!out_value || !a || !b) return false;

    array *output;

    if (array_subtract(&output, a->data, b->data) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_subtract");
        return false;
    }

    bool req_grad = a->requires_grad || b->requires_grad;

    if (value_create_value(out_value, output, req_grad, !req_grad) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
        return false;
    }

    if (req_grad) {
        Function *fn;
        if (function_create_function(&fn, array_subtract_backward) == false) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_create_function");
            return false;
        }
        if (!function_append_input(fn, a)) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_append_input");
            return false;
        }
        if (!function_append_input(fn, b)) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_append_input");
            return false;
        }
        if (value_set_creator(*out_value, fn) == false) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_set_creator");
            return false;
        }
    }

    return true;
}

bool value_multiply(Value **out_value, Value *a, Value *b) {

    if (!out_value || !a || !b) return false;

    array *output;

    if (array_multiply(&output, a->data, b->data) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_multiply");
        return false;
    }

    bool req_grad = a->requires_grad || b->requires_grad;
    if (value_create_value(out_value, output, req_grad, !req_grad) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
        return false;
    }

    if (req_grad) {
        Function *fn;
        if (function_create_function(&fn, array_multiply_backward) == false) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_create_function");
            return false;
        }
        if (!function_append_input(fn, a)) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_append_input");
            return false;
        }
        if (!function_append_input(fn, b)) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_append_input");
            return false;
        }
        if (value_set_creator(*out_value, fn) == false) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_set_creator");
            return false;
        }
    }

    return true;
}

bool value_divide(Value **out_value, Value *a, Value *b) {

    if (!out_value || !a || !b) return false;

    array *output;
    if (array_divide(&output, a->data, b->data) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_divide");
        return false;
    }

    bool req_grad = a->requires_grad || b->requires_grad;
    if (value_create_value(out_value, output, req_grad, !req_grad) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
        return false;
    }

    if (req_grad) {
        Function *fn;
        if (function_create_function(&fn, array_divide_backward) == false) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_create_function");
            return false;
        }
        if (!function_append_input(fn, a)) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_append_input");
            return false;
        }
        if (!function_append_input(fn, b)) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_append_input");
            return false;
        }
        if (value_set_creator(*out_value, fn) == false) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_set_creator");
            return false;
        }
    }

    return true;
}

bool value_floor_divide(Value **out_value, Value *a, Value *b) {

    if (!out_value || !a || !b) return false;

    array *output;

    if (array_floor_divide(&output, a->data, b->data) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_floor_divide");
        return false;
    }

    bool req_grad = a->requires_grad || b->requires_grad;
    if (value_create_value(out_value, output, req_grad, !req_grad) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
        return false;
    }

    // if (req_grad) {
    //     Function *fn;
    //     if (function_create_function(&fn, array_add_backward) == false) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_create_function");
    //         return false;
    //     }
    //     if (!function_append_input(fn, a)) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_append_input");
    //         return false;
    //     }
    //     if (!function_append_input(fn, b)) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_append_input");
    //         return false;
    //     }
    //     if (value_set_creator(*out_value, fn) == false) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_set_creator");
    //         return false;
    //     }
    // }

    return true;
}

bool value_power(Value **out_value, Value *a, Value *b) {

    if (!out_value || !a || !b) return false;

    array *output;

    if (array_power(&output, a->data, b->data) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_power");
        return false;
    }

    bool req_grad = a->requires_grad || b->requires_grad;
    if (value_create_value(out_value, output, req_grad, !req_grad) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
        return false;
    }

    if (req_grad) {
        Function *fn;
        if (function_create_function(&fn, array_power_backward) == false) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_create_function");
            return false;
        }
        if (!function_append_input(fn, a)) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_append_input");
            return false;
        }
        if (!function_append_input(fn, b)) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_append_input");
            return false;
        }
        if (value_set_creator(*out_value, fn) == false) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_set_creator");
            return false;
        }
    }

    return true;
}



bool value_equality(Value **out_value, Value *a, Value *b) {

    if (!out_value || !a || !b) return false;

    array *output;
    if (array_equality(&output, a->data, b->data) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_equality");
        return false;
    }

    if (value_create_value(out_value, output, false, true) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
        return false;
    }

    return true;
}

bool value_greater_than(Value **out_value, Value *a, Value *b) {

    if (!out_value || !a || !b) return false;

    array *output;
    if (array_greater_than(&output, a->data, b->data) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_greater_than");
        return false;
    }

    if (value_create_value(out_value, output, false, true) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
        return false;
    }

    return true;
}

bool value_less_than(Value **out_value, Value *a, Value *b) {

    if (!out_value || !a || !b) return false;

    array *output;
    if (array_less_than(&output, a->data, b->data) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_less_than");
        return false;
    }

    if (value_create_value(out_value, output, false, true) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
        return false;
    }

    return true;
}

bool value_maximum(Value **out_value, Value *a, Value *b) {

    if (!out_value || !a || !b) return false;

    array *output;
    if (array_maximum(&output, a->data, b->data) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_maximum");
        return false;
    }

    bool req_grad = a->requires_grad || b->requires_grad;
    if (value_create_value(out_value, output, req_grad, !req_grad) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
        return false;
    }

    // if (req_grad) {
    //     Function *fn;
    //     if (function_create_function(&fn, array_add_backward) == false) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_create_function");
    //         return false;
    //     }
    //     if (!function_append_input(fn, a)) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_append_input");
    //         return false;
    //     }
    //     if (!function_append_input(fn, b)) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_append_input");
    //         return false;
    //     }
    //     if (value_set_creator(*out_value, fn) == false) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_set_creator");
    //         return false;
    //     }
    // }

    return true;
}

bool value_minimum(Value **out_value, Value *a, Value *b) {

    if (!out_value || !a || !b) return false;

    array *output;
    if (array_minimum(&output, a->data, b->data) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_minimum");
        return false;
    }

    bool req_grad = a->requires_grad || b->requires_grad;
    if (value_create_value(out_value, output, req_grad, !req_grad) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
        return false;
    }

    // if (req_grad) {
    //     Function *fn;
    //     if (function_create_function(&fn, array_add_backward) == false) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_create_function");
    //         return false;
    //     }
    //     if (!function_append_input(fn, a)) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_append_input");
    //         return false;
    //     }
    //     if (!function_append_input(fn, b)) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_append_input");
    //         return false;
    //     }
    //     if (value_set_creator(*out_value, fn) == false) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_set_creator");
    //         return false;
    //     }
    // }

    return true;
}



bool value_square_root(Value **out_value, Value *a) {
   
    if (!out_value || !a) return false;

    array *output;

    if (array_square_root(&output, a->data) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_square_root");
        return false;
    }

    bool req_grad = a->requires_grad;
    if (value_create_value(out_value, output, req_grad, !req_grad) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
        return false;
    }

    // if (req_grad) {
    //     Function *fn;
    //     if (function_create_function(&fn, array_add_backward) == false) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_create_function");
    //         return false;
    //     }
    //     if (!function_append_input(fn, a)) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_append_input");
    //         return false;
    //     }
    //     if (value_set_creator(*out_value, fn) == false) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_set_creator");
    //         return false;
    //     }
    // }

    return true;
}

bool value_absolute_value(Value **out_value, Value *a) {
    
    if (!out_value || !a) return false;

    array *output;

    if (array_absolute_value(&output, a->data) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_absolute_value");
        return false;
    }

    bool req_grad = a->requires_grad;
    if (value_create_value(out_value, output, req_grad, !req_grad) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
        return false;
    }

    // if (req_grad) {
    //     Function *fn;
    //     if (function_create_function(&fn, array_add_backward) == false) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_create_function");
    //         return false;
    //     }
    //     if (!function_append_input(fn, a)) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_append_input");
    //         return false;
    //     }
    //     if (value_set_creator(*out_value, fn) == false) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_set_creator");
    //         return false;
    //     }
    // }

    return true;
}

bool value_negation(Value **out_value, Value *a) {

    if (!out_value || !a) return false;

    array *output;

    if (array_negation(&output, a->data) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_negation");
        return false;
    }

    bool req_grad = a->requires_grad;
    if (value_create_value(out_value, output, req_grad, !req_grad) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
        return false;
    }

    // if (req_grad) {
    //     Function *fn;
    //     if (function_create_function(&fn, array_add_backward) == false) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_create_function");
    //         return false;
    //     }
    //     if (!function_append_input(fn, a)) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_append_input");
    //         return false;
    //     }
    //     if (value_set_creator(*out_value, fn) == false) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_set_creator");
    //         return false;
    //     }
    // }

    return true;
}



bool value_sum(Value **out_value, Value *a, uint64_t rndim, uint64_t *dims, bool keepdim) {
    
    if (!out_value || !a) return false;

    array *output;

    if (array_sum(&output, a->data, rndim, dims, keepdim) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_sum");
        return false;
    }

    bool req_grad = a->requires_grad;
    if (value_create_value(out_value, output, req_grad, !req_grad) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
        return false;
    }

    // if (req_grad) {
    //     Function *fn;
    //     if (function_create_function(&fn, array_add_backward) == false) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_create_function");
    //         return false;
    //     }
    //     if (!function_append_input(fn, a)) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_append_input");
    //         return false;
    //     }
    //     if (value_set_creator(*out_value, fn) == false) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_set_creator");
    //         return false;
    //     }
    // }

    return true;
}

bool value_max(Value **out_value, Value *a, uint64_t rndim, uint64_t *dims, bool keepdim) {
    
    if (!out_value || !a) return false;

    array *output;

    if (array_max(&output, a->data, rndim, dims, keepdim) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_max");
        return false;
    }

    bool req_grad = a->requires_grad;
    if (value_create_value(out_value, output, req_grad, !req_grad) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
        return false;
    }

    // if (req_grad) {
    //     Function *fn;
    //     if (function_create_function(&fn, array_add_backward) == false) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_create_function");
    //         return false;
    //     }
    //     if (!function_append_input(fn, a)) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_append_input");
    //         return false;
    //     }
    //     if (value_set_creator(*out_value, fn) == false) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_set_creator");
    //         return false;
    //     }
    // }

    return true;
}

bool value_min(Value **out_value, Value *a, uint64_t rndim, uint64_t *dims, bool keepdim) {
    
    if (!out_value || !a) return false;

    array *output;

    if (array_min(&output, a->data, rndim, dims, keepdim) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_min");
        return false;
    }

    bool req_grad = a->requires_grad;
    if (value_create_value(out_value, output, req_grad, !req_grad) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
        return false;
    }

    // if (req_grad) {
    //     Function *fn;
    //     if (function_create_function(&fn, array_add_backward) == false) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_create_function");
    //         return false;
    //     }
    //     if (!function_append_input(fn, a)) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_append_input");
    //         return false;
    //     }
    //     if (value_set_creator(*out_value, fn) == false) {
    //         REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_set_creator");
    //         return false;
    //     }
    // }

    return true;
}

bool value_matmul(Value **out_value, Value *a, Value *b) {
    
    if (!out_value || !a || !b) return false;

    array *output;
    
    if (array_matmul(&output, a->data, b->data) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_matmul");
        return false;
    }

    bool req_grad = a->requires_grad || b->requires_grad;
    if (value_create_value(out_value, output, req_grad, !req_grad) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
        return false;
    }

    if (req_grad) {
        REPORT_ERROR(NOT_IMPLEMENTED, "matmul does not have a backward function currently");
        // Function *fn;
        // if (function_create_function(&fn, array_matmul_backward) == false) {
        //     REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_create_function");
        //     return false;
        // }
        // if (!function_append_input(fn, a)) {
        //     REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_append_input");
        //     return false;
        // }
        // if (!function_append_input(fn, b)) {
        //     REPORT_ERROR(ERR_FUNC_CALL_FAILED, "function_append_input");
        //     return false;
        // }
        // if (value_set_creator(*out_value, fn) == false) {
        //     REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_set_creator");
        //     return false;
        // }
    }

    return true;
}
