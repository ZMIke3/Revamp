#include "Functions.h"


bool array_zeros(array **out_array, uint64_t ndim, uint64_t *shape, dtype type, Backend backend) {
    OpInstr instr = {
        .op = OP_ZEROS,
        .reduction_op = false,
        .backward_pass = false,
        .creation_op = true,
        .creation = {
            .ndim = ndim,
            .shape = shape,
            .dtype = type,
            .backend = backend
        }
    };

    ArrayList *list = NULL;

    OperationContext *ctx;
    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   *out_array = ctx->output;

   if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    return true;
}

bool array_zeros_like(array *in, array **out_array) {
    OpInstr instr = { 
        .op = OP_ZEROS,
        .reduction_op = false,
        .backward_pass = false,
        .creation_op = true,
        .creation = {
            .ndim = in->ndim,
            .shape = in->shape,
            .dtype = in->type_obj->type,
            .backend = in->backend
        }
    };

    ArrayList *list = NULL;

    OperationContext *ctx;
    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   *out_array = ctx->output;

   if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    return true;
}

bool array_ones(array **out_array, uint64_t ndim, uint64_t *shape, dtype type, Backend backend) {
    OpInstr instr = {
        .op = OP_ONES,
        .reduction_op = false,
        .backward_pass = false,
        .creation_op = true,
        .creation = {
            .ndim = ndim,
            .shape = shape,
            .dtype = type,
            .backend = backend
        }
    };

    ArrayList *list = NULL;

    OperationContext *ctx;
    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   *out_array = ctx->output;

   if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    return true;
}

bool array_ones_like(array *in, array **out_array) {
    OpInstr instr = {
        .op = OP_ONES,
        .reduction_op = false,
        .backward_pass = false,
        .creation_op = true,
        .creation = {
            .ndim = in->ndim,
            .shape = in->shape,
            .dtype = in->type_obj->type,
            .backend = in->backend
        }
    };

    ArrayList *list = NULL;

    OperationContext *ctx;
    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   *out_array = ctx->output;

   if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    return true;
}

bool array_fill(array **out_array, uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double fill_value) {
    OpInstr instr = {  
        .op = OP_FILL,
        .reduction_op = false,
        .backward_pass = false,
        .creation_op = true,
        .creation = {
            .ndim = ndim,
            .shape = shape,
            .dtype = type,
            .backend = backend,
            .fill_value = fill_value
        }
    };

    ArrayList *list = NULL;

    OperationContext *ctx;
    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   *out_array = ctx->output;

   if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    return true;
}

bool array_arange(array **out_array, uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double start, double step) {
    OpInstr instr = {
        .op = OP_ARANGE,
        .reduction_op = false,
        .backward_pass = false,
        .creation_op = true,
        .creation = {
            .ndim = ndim,
            .shape = shape,
            .dtype = type,
            .backend = backend,
            .start = start,
            .step = step
        }
    };

    ArrayList *list = NULL;

    OperationContext *ctx;
    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   *out_array = ctx->output;

   if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    return true;
}

bool array_linspace(array **out_array, uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double start, double stop) {
    OpInstr instr = {
        .op = OP_LINSPACE,
        .reduction_op = false,
        .backward_pass = false,
        .creation_op = true,
        .creation = {
            .ndim = ndim,
            .shape = shape,
            .dtype = type,
            .backend = backend,
            .start = start,
            .stop = stop
        }
    };

    ArrayList *list = NULL;

    OperationContext *ctx;
    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   *out_array = ctx->output;

   if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    return true;
}

bool array_randu(array **out_array, uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double low, double high) {
    OpInstr instr = {
        .op = OP_RANDU,
        .reduction_op = false,
        .backward_pass = false,
        .creation_op = true,
        .creation = {
            .ndim = ndim,
            .shape = shape,
            .dtype = type,
            .backend = backend,
            .low = low,
            .high = high
        }

    };

    ArrayList *list = NULL;

    OperationContext *ctx;
    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   *out_array = ctx->output;

   if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    return true;
}

bool array_randn(array **out_array, uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double mean, double stdev) {
    OpInstr instr = {
        .op = OP_RANDN,
        .reduction_op = false,
        .backward_pass = false,
        .creation_op = true,
        .creation = {
            .ndim = ndim,
            .shape = shape,
            .dtype = type,
            .backend = backend,
            .mean = mean,
            .stdev = stdev
        }
    };

    ArrayList *list = NULL;

    OperationContext *ctx;
    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   *out_array = ctx->output;

   if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    return true;
}

bool array_randint(array **out_array, uint64_t ndim, uint64_t *shape, dtype type, Backend backend, double low, double high) {
    OpInstr instr = { 
        .op = OP_RANDINT,
        .reduction_op = false,
        .backward_pass = false,
        .creation_op = true,
        .creation = {
            .ndim = ndim,
            .shape = shape,
            .dtype = type,
            .backend = backend,
            .low = low,
            .high = high
        }
    };

    ArrayList *list = NULL;

    OperationContext *ctx;
    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   *out_array = ctx->output;

   if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    return true;
}


bool array_add(array **out_array, array *a, array *b) {
    OpInstr instr = {
        .op = OP_ADD,
        .inArity = 2,
        .outArity = 1,
        .reduction_op = false,
        .backward_pass = false,
        .creation_op = false,
    };

    ArrayList *list;

    if (arraylist_create(&list) == false) {
        printf("1.1\n");
        return false;
    }

    if (!arraylist_append(list, a)) {  
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    if (!arraylist_append(list, b)) {  
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    OperationContext *ctx;
    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   if (ctx->iter->run(ctx->iter, ctx->fn) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: Iter_run_kernel");
        return false;
   }

   *out_array = ctx->output;

   if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

   if (arraylist_free(list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_free");
        return false;
   }

   return true;

}

bool array_subtract(array **out_array, array *a, array *b) {
    OpInstr instr = {
        .op = OP_SUBTRACT,
        .inArity = 2,
        .outArity = 1,
        .reduction_op = false,
        .backward_pass = false,
        .creation_op = false,
    };

    ArrayList *list;
    if (arraylist_create(&list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_create");
        return false;
    }

    if (!arraylist_append(list, a)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }
    if (!arraylist_append(list, b)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    OperationContext *ctx;
    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   if (ctx->iter->run(ctx->iter, ctx->fn) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: Iter_run_kernel");
        return false;
   }

    *out_array = ctx->output;

    if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    if (arraylist_free(list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_free");
        return false;
    }

    return true;
}

bool array_multiply(array **out_array, array *a, array *b) {
    OpInstr instr = {
        .op = OP_MULTIPLY,
        .inArity = 2,
        .outArity = 1,
        .reduction_op = false,
        .backward_pass = false,
        .creation_op = false,
    };

    ArrayList *list;
    if (arraylist_create(&list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_create");
        return false;
    }

    if (!arraylist_append(list, a)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }
    if (!arraylist_append(list, b)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    OperationContext *ctx;
    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   if (ctx->iter->run(ctx->iter, ctx->fn) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: Iter_run_kernel");
        return false;
   }

    *out_array = ctx->output;

    if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    if (arraylist_free(list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_free");
        return false;
    }

    return true;
}

bool array_divide(array **out_array, array *a, array *b) {
    OpInstr instr = {
        .op = OP_DIVIDE,
        .inArity = 2,
        .outArity = 1,
        .reduction_op = false,
        .backward_pass = false,
        .creation_op = false,
        .force_cast_inputs = true,
        .cast_to = DTYPE_F64
    };

    ArrayList *list;
    if (arraylist_create(&list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_create");
        return false;
    }

    if (!arraylist_append(list, a)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }
    if (!arraylist_append(list, b)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    OperationContext *ctx;
    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   if (ctx->iter->run(ctx->iter, ctx->fn) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: Iter_run_kernel");
        return false;
   }

    *out_array = ctx->output;

    if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    if (arraylist_free(list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_free");
        return false;
    }

    return true;
}

bool array_floor_divide(array **out_array, array *a, array *b) {
    OpInstr instr = {
        .op = OP_DIVIDE,
        .inArity = 2,
        .outArity = 1,
        .reduction_op = false,
        .backward_pass = false,
        .creation_op = false,
        .force_cast_inputs = true,
        .cast_to = DTYPE_F32
    };

    ArrayList *list;
    if (arraylist_create(&list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_create");
        return false;
    }

    if (!arraylist_append(list, a)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }
    if (!arraylist_append(list, b)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    OperationContext *ctx;
    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   if (ctx->iter->run(ctx->iter, ctx->fn) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: Iter_run_kernel");
        return false;
   }

    *out_array = ctx->output;

    if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    if (arraylist_free(list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_free");
        return false;
    }

    return true;
}

bool array_power(array **out_array, array *a, array *b) {
    OpInstr instr = {
        .op = OP_POWER,
        .inArity = 2,
        .outArity = 1,
        .reduction_op = false,
        .backward_pass = false,
        .creation_op = false,
        .force_cast_inputs = true,
        .cast_to = DTYPE_F64,
    };

    ArrayList *list;
    if (arraylist_create(&list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_create");
        return false;
    }

    if (!arraylist_append(list, a)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }
    if (!arraylist_append(list, b)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    OperationContext *ctx;
    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   if (ctx->iter->run(ctx->iter, ctx->fn) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: Iter_run_kernel");
        return false;
   }

    *out_array = ctx->output;

    if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    if (arraylist_free(list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_free");
        return false;
    }

    return true;
}



bool array_equality(array **out_array, array *a, array *b) {
    OpInstr instr = {
        .op = OP_EQUALITY,
        .inArity = 2,
        .outArity = 1,
        .reduction_op = false,
        .backward_pass = false,
        .creation_op = false,
    };

    ArrayList *list;
    if (arraylist_create(&list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_create");
        return false;
    }

    if (!arraylist_append(list, a)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }
    if (!arraylist_append(list, b)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    OperationContext *ctx;
    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   if (ctx->iter->run(ctx->iter, ctx->fn) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: Iter_run_kernel");
        return false;
   }

    *out_array = ctx->output;

    if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    if (arraylist_free(list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_free");
        return false;
    }

    return true;
}

bool array_greater_than(array **out_array, array *a, array *b) {
    OpInstr instr = {
        .op = OP_GREATER_THAN,
        .inArity = 2,
        .outArity = 1,
        .reduction_op = false,
        .backward_pass = false,
        .creation_op = false,
    };

    ArrayList *list;
    if (arraylist_create(&list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_create");
        return false;
    }

    if (!arraylist_append(list, a)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }
    if (!arraylist_append(list, b)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    OperationContext *ctx;
    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   if (ctx->iter->run(ctx->iter, ctx->fn) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: Iter_run_kernel");
        return false;
   }

    *out_array = ctx->output;

    if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    if (arraylist_free(list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_free");
        return false;
    }

    return true;
}

bool array_less_than(array **out_array, array *a, array *b) {
    OpInstr instr = {
        .op = OP_LESS_THAN,
        .inArity = 2,
        .outArity = 1,
        .reduction_op = false,
        .backward_pass = false,
        .creation_op = false,
    };

    ArrayList *list;
    if (arraylist_create(&list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_create");
        return false;
    }

    if (!arraylist_append(list, a)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }
    if (!arraylist_append(list, b)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    OperationContext *ctx;
    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   if (ctx->iter->run(ctx->iter, ctx->fn) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: Iter_run_kernel");
        return false;
   }

    *out_array = ctx->output;

    if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    if (arraylist_free(list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_free");
        return false;
    }

    return true;
}

bool array_maximum(array **out_array, array *a, array *b) {
    OpInstr instr = {
        .op = OP_ELEMENT_WISE_MAXIMUM,
        .inArity = 2,
        .outArity = 1,
        .reduction_op = false,
        .backward_pass = false,
        .creation_op = false,
    };

    ArrayList *list;
    if (arraylist_create(&list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_create");
        return false;
    }

    if (!arraylist_append(list, a)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }
    if (!arraylist_append(list, b)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    OperationContext *ctx;
    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   if (ctx->iter->run(ctx->iter, ctx->fn) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: Iter_run_kernel");
        return false;
   }

    *out_array = ctx->output;

    if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    if (arraylist_free(list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_free");
        return false;
    }

    return true;
}

bool array_minimum(array **out_array, array *a, array *b) {
    OpInstr instr = {
        .op = OP_ELEMENT_WISE_MINIMUM,
        .inArity = 2,
        .outArity = 1,
        .reduction_op = false,
        .backward_pass = false,
        .creation_op = false,
    };

    ArrayList *list;
    if (arraylist_create(&list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_create");
        return false;
    }

    if (!arraylist_append(list, a)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }
    if (!arraylist_append(list, b)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    OperationContext *ctx;
    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   if (ctx->iter->run(ctx->iter, ctx->fn) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: Iter_run_kernel");
        return false;
   }

    *out_array = ctx->output;

    if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    if (arraylist_free(list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_free");
        return false;
    }

    return true;
}



bool array_square_root(array **out_array, array *a) {
    OpInstr instr = {
        .op = OP_SQUARE_ROOT,
        .inArity = 1,
        .outArity = 1,
        .reduction_op = false,
        .backward_pass = false,
        .creation_op = false,
        
    };

    ArrayList *list;
    if (arraylist_create(&list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_create");
        return false;
    }

    if (!arraylist_append(list, a)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    OperationContext *ctx;
    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   if (ctx->iter->run(ctx->iter, ctx->fn) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: Iter_run_kernel");
        return false;
   }

    *out_array = ctx->output;

    if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    if (arraylist_free(list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_free");
        return false;
    }

    return true;
}

bool array_absolute_value(array **out_array, array *a) {
    OpInstr instr = {
        .op = OP_ABSOLUTE_VALUE,
        .inArity = 1,
        .outArity = 1,
        .reduction_op = false,
        .backward_pass = false,
        .creation_op = false,
        
    };

    ArrayList *list;
    if (arraylist_create(&list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_create");
        return false;
    }

    if (!arraylist_append(list, a)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    OperationContext *ctx;
    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   if (ctx->iter->run(ctx->iter, ctx->fn) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: Iter_run_kernel");
        return false;
   }

    *out_array = ctx->output;

    if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    if (arraylist_free(list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_free");
        return false;
    }

    return true;
}

bool array_negation(array **out_array, array *a) {
    OpInstr instr = {
        .op = OP_NEGATION,  
        .inArity = 1,
        .outArity = 1,
        .reduction_op = false,
        .backward_pass = false,
        .creation_op = false,
              
    };

    ArrayList *list;
    if (arraylist_create(&list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_create");
        return false;
    }

    if (!arraylist_append(list, a)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    OperationContext *ctx;
    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   if (ctx->iter->run(ctx->iter, ctx->fn) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: Iter_run_kernel");
        return false;
   }

    *out_array = ctx->output;

    if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    if (arraylist_free(list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_free");
        return false;
    }

    return true;
}



bool array_sum(array **out_array, array *a, uint64_t rndim, uint64_t *dims, bool keepdim) {
    OpInstr instr = {
        .op = OP_SUM_REDUCTION,  
        .inArity = 1,
        .outArity = 1,      
        .reduction_op = true,
        .backward_pass = false,
        .creation_op = false,
        .reduction = {
            .input_shape = a->shape,
            .input_ndim = a->ndim,
            .number_reduction_dimensions = rndim,
            .reduction_dimensions = dims,
            .keepdim = keepdim
        }
    };

    ArrayList *list;
    if (arraylist_create(&list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_create");
        return false;
    }

    if (!arraylist_append(list, a)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    OperationContext *ctx;
    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   if (ctx->iter->run(ctx->iter, ctx->fn) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: Iter_run_kernel");
        return false;
   }

    *out_array = ctx->output;

    if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    if (arraylist_free(list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_free");
        return false;
    }

    return true;
}

bool array_max(array **out_array, array *a, uint64_t rndim, uint64_t *dims, bool keepdim) {
    OpInstr instr = {
        .op = OP_MAX_REDUCTION,        
        .inArity = 1,
        .outArity = 1,      
        .reduction_op = true,
        .backward_pass = false,
        .creation_op = false,
        .reduction = {
            .input_shape = a->shape,
            .input_ndim = a->ndim,
            .number_reduction_dimensions = rndim,
            .reduction_dimensions = dims,
            .keepdim = keepdim
        }
    };

    ArrayList *list;
    if (arraylist_create(&list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_create");
        return false;
    }

    if (!arraylist_append(list, a)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    OperationContext *ctx;
    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   if (ctx->iter->run(ctx->iter, ctx->fn) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: Iter_run_kernel");
        return false;
   }

    *out_array = ctx->output;

    if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    if (arraylist_free(list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_free");
        return false;
    }

    return true;
}

bool array_min(array **out_array, array *a, uint64_t rndim, uint64_t *dims, bool keepdim) {
    OpInstr instr = {
        .op = OP_MIN_REDUCTION,
        .inArity = 1,
        .outArity = 1,      
        .reduction_op = true,
        .backward_pass = false,
        .creation_op = false,
        .reduction = {
            .input_shape = a->shape,
            .input_ndim = a->ndim,
            .number_reduction_dimensions = rndim,
            .reduction_dimensions = dims,
            .keepdim = keepdim
        }
    };

    ArrayList *list;
    if (arraylist_create(&list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_create");
        return false;
    }

    if (!arraylist_append(list, a)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    OperationContext *ctx;
    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   if (ctx->iter->run(ctx->iter, ctx->fn) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: Iter_run_kernel");
        return false;
   }

    *out_array = ctx->output;

    if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    if (arraylist_free(list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_free");
        return false;
    }

    return true;
}


bool array_matmul(array **out_array, array *a, array *b) {
    OpInstr instr = {
        .op = OP_MATMUL,
        .reduction_op = false,
        .creation_op = false,  
    };
   // printf("2\n");


    ArrayList *list;

   // printf("3\n");

    if (arraylist_create(&list) == false) {
        return false;
    }

  //  printf("4\n");


    if (!arraylist_append(list, a)) {  
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

   // printf("5\n");


    if (!arraylist_append(list, b)) {  
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

  //  printf("6\n");

    OperationContext *ctx;
    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }


  //  printf("7\n");

   if (ctx->iter->run(ctx->iter, ctx->fn) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: Iter_run_kernel");
        return false;
   }
   // printf("8\n");


   *out_array = ctx->output;


   if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

  //  printf("9\n");


   if (arraylist_free(list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_free");
        return false;
   }

  //  printf("10\n");

    return true;


}


bool array_print_array(array *array) {
    if (!array) {
       REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "");
       return false;
    }
    ArrayList *list;
    if (!arraylist_create(&list)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_create");
        return false;
    }

    if (!arraylist_append(list, array)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    Iter *iter;
    if (!Iter_create_iterator(&iter, list, NULL, ITER_ELEMENTWISE)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: Iter_create_iterator");
        return false;
    }

    ElementwiseIter *e = (ElementwiseIter *) iter;

    for (int i = 0; i < e->ndim; i++) {
        printf("[");
    }

    dtype type = array->type_obj->type;


    for (uint64_t elem = 0; elem < e->total_size; elem++) {
       
        if (elem > 0) {
            for (int i = e->ndim - 1; i >= 0; i--) {
                if (e->coordinates[i] == 0) {
                    printf("[");
                } else {
                    break;
                }
            }
        }

        switch (type) {
            case DTYPE_F32:
                printf("%f", *(float *)e->dataptrs[0]);
            break;
            case DTYPE_F64:
                printf("%f", *(double *)e->dataptrs[0]);
            break;
            case DTYPE_I32:
                printf("%d", *(int *)e->dataptrs[0]);
            break;
            case DTYPE_I64:
                printf("%ld", *(int64_t *)e->dataptrs[0]);
            break;
            case DTYPE_U64:
                printf("%"PRIu64"", *(uint64_t *)e->dataptrs[0]);
            break;
        
        default:
            break;
        }

        if (elem + 1 < e->total_size) {
            bool any_dimension_ending = false;
            for (int i = e->ndim - 1; i >= 0; i--) {
                if (e->coordinates[i] + 1 == e->shape[i]) {
                    printf("]");
                    any_dimension_ending = true;
                } else {
                    break;
                }
            }

            printf(", ");

            if (any_dimension_ending) {
                printf("\n");
                int indent = 0;
                for (int i = e->ndim - 1; i >= 0; i--) {
                    if (e->coordinates[i] + 1 == e->shape[i]) {
                        indent = i;
                    } else {
                        break;
                    }
                }
                for (int j = 0; j < indent; j++) {
                    printf(" ");
                }
            }

            if (iter->next_ele(iter) == false) {
                REPORT_ERROR(ERR_FUNC_CALL_FAILED, "element_wise_iter_next_ele");
            }
        }
    }

    for (int i = 0; i < e->ndim; i++) {
        printf("]");
    }
    printf("\n\n");

    arraylist_free(list);
}
