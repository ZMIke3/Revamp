#include "Ops.h"

static bool rule_checker(OpType op, dtype type) {

    if (type == DTYPE_NONE || type == DTYPE_COUNT) return false;

    switch (op) {

        case OP_ADD:
        case OP_SUBTRACT:
        case OP_MULTIPLY:
            return type != DTYPE_BOOL;

        case OP_DIVIDE:
        case OP_MATMUL:
            return type == DTYPE_F32 || type == DTYPE_F64;

        case OP_SQUARE_ROOT:
        case OP_LOG:
        case OP_EXP:
        case OP_POWER:
            return type == DTYPE_F32 || type == DTYPE_F64;

        case OP_ABSOLUTE_VALUE:
        case OP_NEGATION:
            return type == DTYPE_I8  || type == DTYPE_I16 ||
                   type == DTYPE_I32 || type == DTYPE_I64 ||
                   type == DTYPE_F32 || type == DTYPE_F64;

        case OP_EQUALITY:
        case OP_GREATER_THAN:
        case OP_LESS_THAN:
        case OP_ELEMENT_WISE_MAXIMUM:
        case OP_ELEMENT_WISE_MINIMUM:
            return type != DTYPE_BOOL;

        case OP_ZEROS:      case OP_ONES:
        case OP_ZEROS_LIKE: case OP_ONES_LIKE:
        case OP_FILL:
        case OP_ARANGE:
        case OP_LINSPACE:
            return type != DTYPE_BOOL;

        case OP_RANDU:
        case OP_RANDN:
            return type == DTYPE_F32 || type == DTYPE_F64;

        case OP_RANDINT:
            return type == DTYPE_I8  || type == DTYPE_I16 ||
                   type == DTYPE_I32 || type == DTYPE_I64 ||
                   type == DTYPE_U8  || type == DTYPE_U16 ||
                   type == DTYPE_U32 || type == DTYPE_U64;

        case OP_SUM_REDUCTION:
        case OP_MAX_REDUCTION:
        case OP_MIN_REDUCTION:
        case OP_ARGMAX_REDUCTION:
        case OP_ARGMIN_REDUCTION:
            return type != DTYPE_BOOL;

        case OP_MEAN_REDUCTION:
            return type == DTYPE_F32 || type == DTYPE_F64;

        case OP_WHERE:
            return true;

        case OP_ADD_BACKWARDS:
        case OP_SUBTRACT_BACKWARDS:
        case OP_MULTIPLY_BACKWARDS:
        case OP_DIVIDE_BACKWARDS:
        case OP_POWER_BACKWARDS:
        case OP_LOG_BACKWARDS:
        case OP_EXP_BACKWARDS:
            return type == DTYPE_F32 || type == DTYPE_F64;

        default:
            return false;
    }
}

static bool op_type_to_kernel_id(OpType op, kernel_id *id) {
    switch (op) {
        case OP_ADD:                  *id = KERNEL_ADD;                  return true;
        case OP_SUBTRACT:             *id = KERNEL_SUBTRACT;             return true;
        case OP_MULTIPLY:             *id = KERNEL_MULTIPLY;             return true;
        case OP_DIVIDE:               *id = KERNEL_DIVIDE;               return true;
        case OP_SQUARE_ROOT:          *id = KERNEL_SQUARE_ROOT;          return true;
        case OP_ABSOLUTE_VALUE:       *id = KERNEL_ABSOLUTE_VALUE;       return true;
        case OP_NEGATION:             *id = KERNEL_NEGATION;             return true;
        case OP_POWER:                *id = KERNEL_POWER;                return true;
        case OP_EQUALITY:             *id = KERNEL_EQUALITY;             return true;
        case OP_GREATER_THAN:         *id = KERNEL_GREATER_THAN;         return true;
        case OP_LESS_THAN:            *id = KERNEL_LESS_THAN;            return true;
        case OP_ELEMENT_WISE_MAXIMUM: *id = KERNEL_ELEMENT_WISE_MAXIMUM; return true;
        case OP_ELEMENT_WISE_MINIMUM: *id = KERNEL_ELEMENT_WISE_MINIMUM; return true;
        case OP_WHERE:                *id = KERNEL_WHERE;                return true;
        case OP_ZEROS:                *id = KERNEL_ZEROS;                return true;
        case OP_ONES:                 *id = KERNEL_ONES;                 return true;
        case OP_FILL:                 *id = KERNEL_FILL;                 return true;
        case OP_SUM_REDUCTION:        *id = KERNEL_SUM_REDUCTION;        return true;
        case OP_MAX_REDUCTION:        *id = KERNEL_MAX_REDUCTION;        return true;
        case OP_MIN_REDUCTION:        *id = KERNEL_MIN_REDUCTION;        return true;
        case OP_LOG:                  *id = KERNEL_LOG;                   return true;
        case OP_EXP:                  *id = KERNEL_EXP;                   return true;
        case OP_ARGMAX_REDUCTION:      *id = KERNEL_ARGMAX_REDUCTION;      return true;
        case OP_ARGMIN_REDUCTION:      *id = KERNEL_ARGMIN_REDUCTION;      return true;
        case OP_ADD_BACKWARDS:        *id = KERNEL_ADD_BACKWARDS;        return true;
        case OP_SUBTRACT_BACKWARDS:   *id = KERNEL_SUBTRACT_BACKWARDS;   return true;
        case OP_MULTIPLY_BACKWARDS:   *id = KERNEL_MULTIPLY_BACKWARDS;   return true;
        case OP_DIVIDE_BACKWARDS:     *id = KERNEL_DIVIDE_BACKWARDS;     return true;
        case OP_POWER_BACKWARDS:      *id = KERNEL_POWER_BACKWARDS;     return true;
        case OP_LOG_BACKWARDS:        *id = KERNEL_LOG_BACKWARDS;     return true;
        case OP_EXP_BACKWARDS:        *id = KERNEL_EXP_BACKWARDS;     return true;
        case OP_MATMUL:               *id = KERNEL_MATMUL;            return true;
        default:                                                      return false;
    }
}

void print_kernel_name(int id) {
    switch (id) {
        case KERNEL_ADD:                   printf("The selected kernel is KERNEL_ADD\n");                   break;
        case KERNEL_SUBTRACT:              printf("The selected kernel is KERNEL_SUBTRACT\n");              break;
        case KERNEL_MULTIPLY:              printf("The selected kernel is KERNEL_MULTIPLY\n");              break;
        case KERNEL_DIVIDE:                printf("The selected kernel is KERNEL_DIVIDE\n");                break;
        case KERNEL_SQUARE_ROOT:           printf("The selected kernel is KERNEL_SQUARE_ROOT\n");           break;
        case KERNEL_ABSOLUTE_VALUE:        printf("The selected kernel is KERNEL_ABSOLUTE_VALUE\n");        break;
        case KERNEL_NEGATION:              printf("The selected kernel is KERNEL_NEGATION\n");              break;
        case KERNEL_POWER:                 printf("The selected kernel is KERNEL_POWER\n");                 break;
        case KERNEL_EQUALITY:              printf("The selected kernel is KERNEL_EQUALITY\n");              break;
        case KERNEL_GREATER_THAN:          printf("The selected kernel is KERNEL_GREATER_THAN\n");          break;
        case KERNEL_LESS_THAN:             printf("The selected kernel is KERNEL_LESS_THAN\n");             break;
        case KERNEL_ELEMENT_WISE_MAXIMUM:  printf("The selected kernel is KERNEL_ELEMENT_WISE_MAXIMUM\n");  break;
        case KERNEL_ELEMENT_WISE_MINIMUM:  printf("The selected kernel is KERNEL_ELEMENT_WISE_MINIMUM\n");  break;
        case KERNEL_WHERE:                 printf("The selected kernel is KERNEL_WHERE\n");                 break;
        case KERNEL_ZEROS:                 printf("The selected kernel is KERNEL_ZEROS\n");                 break;
        case KERNEL_ONES:                  printf("The selected kernel is KERNEL_ONES\n");                  break;
        case KERNEL_FILL:                  printf("The selected kernel is KERNEL_FILL\n");                  break;
        case KERNEL_SUM_REDUCTION:         printf("The selected kernel is KERNEL_SUM_REDUCTION\n");         break;
        case KERNEL_MAX_REDUCTION:         printf("The selected kernel is KERNEL_MAX_REDUCTION\n");         break;
        case KERNEL_MIN_REDUCTION:         printf("The selected kernel is KERNEL_MIN_REDUCTION\n");         break;
        case KERNEL_ADD_BACKWARDS:         printf("The selected kernel is KERNEL_ADD_BACKWARDS\n");         break;
        default:                           printf("The selected kernel is UNKNOWN_KERNEL (%d)\n", id);       break;
    }
}

static bool identity_get_identity_for_op(OpType op, IdentityType *type) {
    switch (op) {
        case OP_SUM_REDUCTION:
        case OP_MEAN_REDUCTION:
        case OP_ARGMAX_REDUCTION:
            *type = IDENTITY_ZERO;
            return true;
        
        // case OP_PROD_REDUCTION:
        //     *type = IDENTITY_ONE;
        //     return true;
        
        case OP_MIN_REDUCTION:
            *type = IDENTITY_MAX;
            return true;
        
        case OP_MAX_REDUCTION:
            *type = IDENTITY_MIN;
            return true;
        
        default:
            return false;
    }
}

bool helper_buffer_with_identity(void **buffer, dtype type, uint64_t size, IdentityType identity) {
    switch (type) {
        case DTYPE_F64: {
            double *buf = (double *)*buffer;
            double val;
            switch (identity) {
                case IDENTITY_ZERO: val = 0.0; break;
                case IDENTITY_ONE:  val = 1.0; break;
                case IDENTITY_MIN:  val = -DBL_MAX; break;
                case IDENTITY_MAX:  val = DBL_MAX; break;
            }
            for (uint64_t i = 0; i < size; i++) buf[i] = val;
            break;
        }
        
        case DTYPE_F32: {
            float *buf = (float *)*buffer;
            float val;
            switch (identity) {
                case IDENTITY_ZERO: val = 0.0f; break;
                case IDENTITY_ONE:  val = 1.0f; break;
                case IDENTITY_MIN:  val = -FLT_MAX; break;
                case IDENTITY_MAX:  val = FLT_MAX; break;
            }
            for (uint64_t i = 0; i < size; i++) buf[i] = val;
            break;
        }
        
        case DTYPE_I64: {
            int64_t *buf = (int64_t *)*buffer;
            int64_t val;
            switch (identity) {
                case IDENTITY_ZERO: val = 0; break;
                case IDENTITY_ONE:  val = 1; break;
                case IDENTITY_MIN:  val = INT64_MIN; break;
                case IDENTITY_MAX:  val = INT64_MAX; break;
            }
            for (uint64_t i = 0; i < size; i++) buf[i] = val;
            break;
        }
        
        case DTYPE_I32: {
            int32_t *buf = (int32_t *)*buffer;
            int32_t val;
            switch (identity) {
                case IDENTITY_ZERO: val = 0; break;
                case IDENTITY_ONE:  val = 1; break;
                case IDENTITY_MIN:  val = INT32_MIN; break;
                case IDENTITY_MAX:  val = INT32_MAX; break;
            }
            for (uint64_t i = 0; i < size; i++) buf[i] = val;
            break;
        }
        
        case DTYPE_I16: {
            int16_t *buf = (int16_t *)*buffer;
            int16_t val;
            switch (identity) {
                case IDENTITY_ZERO: val = 0; break;
                case IDENTITY_ONE:  val = 1; break;
                case IDENTITY_MIN:  val = INT16_MIN; break;
                case IDENTITY_MAX:  val = INT16_MAX; break;
            }
            for (uint64_t i = 0; i < size; i++) buf[i] = val;
            break;
        }
        
        case DTYPE_I8: {
            int8_t *buf = (int8_t *)*buffer;
            int8_t val;
            switch (identity) {
                case IDENTITY_ZERO: val = 0; break;
                case IDENTITY_ONE:  val = 1; break;
                case IDENTITY_MIN:  val = INT8_MIN; break;
                case IDENTITY_MAX:  val = INT8_MAX; break;
            }
            for (uint64_t i = 0; i < size; i++) buf[i] = val;
            break;
        }
        
        case DTYPE_U32: {
            uint32_t *buf = (uint32_t *)*buffer;
            uint32_t val;
            switch (identity) {
                case IDENTITY_ZERO: val = 0; break;
                case IDENTITY_ONE:  val = 1; break;
                case IDENTITY_MIN:  val = 0; break;
                case IDENTITY_MAX:  val = UINT32_MAX; break;
            }
            for (uint64_t i = 0; i < size; i++) buf[i] = val;
            break;
        }
        
        case DTYPE_U16: {
            uint16_t *buf = (uint16_t *)*buffer;
            uint16_t val;
            switch (identity) {
                case IDENTITY_ZERO: val = 0; break;
                case IDENTITY_ONE:  val = 1; break;
                case IDENTITY_MIN:  val = 0; break;
                case IDENTITY_MAX:  val = UINT16_MAX; break;
            }
            for (uint64_t i = 0; i < size; i++) buf[i] = val;
            break;
        }
        
        case DTYPE_U8: {
            uint8_t *buf = (uint8_t *)*buffer;
            uint8_t val;
            switch (identity) {
                case IDENTITY_ZERO: val = 0; break;
                case IDENTITY_ONE:  val = 1; break;
                case IDENTITY_MIN:  val = 0; break;
                case IDENTITY_MAX:  val = UINT8_MAX; break;
            }
            for (uint64_t i = 0; i < size; i++) buf[i] = val;
            break;
        }
        
        default:
            return false;
    }
    
    return true;
}

static bool reduction_helper_output(OpType op, array **output, uint64_t ndim, uint64_t *shape, size_t size, dtype type, Backend backend) {
    array *out = *output;
    switch (op) {
        case OP_SUM_REDUCTION:
            if (array_create_array(&out, type, ndim, shape, backend) == false) {
                return false;
            }

            if (array_add_buffer(out) == false) {
                return false;
            }
            
            if (helper_buffer_of_zeros((void **)&out->data, size, out->size) == false) {
                return false;
            }

        break;

        case OP_MAX_REDUCTION:
            
    
        default:
        break;
    }


}

static bool configure_operation_elementwise(OperationContext **ctx, OpInstr *instr, ArrayList *list) {
    if (!instr || !list) { return false;}

    *ctx = calloc(1, sizeof(OperationContext));
    if ((*ctx) == NULL) {  return false; }
    (*ctx)->instr = instr;
    (*ctx)->inputs = list;

    dtype common_type;
    Backend common_backend;
    uint64_t *shape;
    uint64_t ndim;
    kernel_fn fn;
    kernel_id id;
    Iter *iter;
    array *output;


    if (!instr->force_cast_inputs) {
        
        if (arraylist_cast_to_common_dtype(list) == false) {
            printf("c2\n");
            free(ctx);
            return false;
        }
    } else {

        if (arraylist_cast_to_dtype(list, instr->cast_to) == false) {
            printf("c3\n");
            free(ctx);
            return false;
        }

    }

    if (arraylist_get_common_dtype(list, &common_type) == false) {
        printf("c4\n");
        free(ctx);
        return false;
    }

    if (arraylist_get_common_backend(list, &common_backend) == false) {
        printf("c5\n");
        free(ctx);
        return false;
    }

    if (arraylist_broadcast_and_max_ndim(list, &shape, &ndim) == false) {
        printf("c6\n");
        free(ctx);
        return false;
    }

    if (array_create_array(&output, common_type, ndim, shape, common_backend) == false) {
        printf("c7\n");
        free(ctx);
        return false;
    }

    if (array_add_buffer(output) == false) {
        printf("c8\n");
        free(ctx);
        free(shape);
        array_dec_refcount(output);
        return false;
    }

    if (arraylist_append(list, output) == false) {
        printf("c9\n");
        free(ctx);
        free(shape);
        array_dec_refcount(output);
        return false;
    }

    if (op_type_to_kernel_id(instr->op, &id) == false) {
        printf("c10\n");
        free(ctx);
        free(shape);
        array_dec_refcount(output);
        return false;
    }

   printf("The kernel id is %d\n", id);

    if (dispatch_get_kernel(&fn, common_type, common_backend, id) == false) {
        printf("c11\n");
        free(ctx);
        free(shape);
        array_dec_refcount(output);
        return false;
    }

    IterCtx i_ctx = {
        .input_shape = NULL,
        .reduction_perm =  NULL,
        .len_of_inner_ndim = 1
    };

    if (!Iter_create_iterator(&iter, list, &i_ctx, ITER_ELEMENTWISE)) {
        printf("c12\n");
        free(ctx);
        free(shape);
        array_dec_refcount(output);
        return false;
    }

    (*ctx)->output = output;
    (*ctx)->iter = iter;
    (*ctx)->fn = fn;
    return true;

}

static bool configure_operation_creation(OperationContext **ctx, OpInstr *instr, ArrayList *list) {
    if (!instr) { return false;}

    *ctx = calloc(1, sizeof(OperationContext));
    if ((*ctx) == NULL) { printf("c1\n"); return false; }

    (*ctx)->instr = instr;

    if (!instr->creation.shape) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "");
        return false;
    }

    if (instr->creation.ndim < 0 || instr->creation.dtype > DTYPE_COUNT || instr->creation.backend > BACKEND_COUNT) {
        REPORT_ERROR(ERR_INPUT_OUT_OF_BOUNDS, "");
        return false;
    }

    uint64_t *shape = instr->creation.shape;
    uint64_t ndim = instr->creation.ndim;
    dtype type = instr->creation.dtype;
    Backend backend = instr->creation.backend;
    double value = instr->creation.fill_value;
    double start = instr->creation.start;
    double step = instr->creation.step;
    double stop = instr->creation.stop;
    double low = instr->creation.low;
    double high = instr->creation.high;
    double mean = instr->creation.mean;
    double stdev = instr->creation.stdev;
    array *output;

    if (array_create_array(&output, type, ndim, shape, backend) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: array_create_array");
        free(ctx);
        return false;
    }

    if (array_add_buffer(output) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: array_add_buffer");
        free(ctx);
        array_dec_refcount(output);
        return false;
    }

    switch (instr->op) {
        case OP_ZEROS:
        case OP_ZEROS_LIKE:
            if (helper_buffer_of_zeros((void**)&(output)->data, (output)->elesize_in_bytes, (output)->size) == false) {
                REPORT_ERROR(ERR_FUNC_CALL_FAILED, "helper_buffer_of_zeros");
                array_dec_refcount(output);
                return false;
            }
            break;

        case OP_ONES:
        case OP_ONES_LIKE:
            if (helper_buffer_of_ones((void**)&(output)->data, (output)->elesize_in_bytes, (output)->size, type) == false) {
                REPORT_ERROR(ERR_FUNC_CALL_FAILED, "helper_buffer_of_ones");
                array_dec_refcount(output);
                return false;
            }
            break;
        
        case OP_FILL:
            if (helper_buffer_with_value((void**)&(output)->data, (output)->elesize_in_bytes, (output)->size, type, value) == false) {
                REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: helper_buffer_with_value");
                array_dec_refcount(output);
                return false;
            }
            break;

        case OP_ARANGE:
            if (helper_buffer_arange((void**)&(output)->data, (output)->elesize_in_bytes, (output)->size, type, start, step) == false) {
                REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: helper_buffer_arange");
                array_dec_refcount(output);
                return false;
            }
            break;

        case OP_LINSPACE:
            if (helper_buffer_linspace((void**)&(output)->data, (output)->elesize_in_bytes, (output)->size, type, start, stop) == false) {
                REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: helper_buffer_linspace");
                array_dec_refcount(output);
                return false;
            }
            break;

        case OP_RANDU:
            if (helper_buffer_randu((void**)&(output)->data, (output)->elesize_in_bytes, (output)->size, type, low, high) == false) {
                REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: helper_buffer_rand");
                array_dec_refcount(output);
                return false;
            }
            break;

        case OP_RANDN:
            if (helper_buffer_randn((void**)&(output)->data, (output)->elesize_in_bytes, (output)->size, type, mean, stdev) == false) {
                REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: helper_buffer_randn");
                array_dec_refcount(output);
                return false;
            }
            break;
        
        case OP_RANDINT:
            if (helper_buffer_randint((void**)&(output)->data, (output)->elesize_in_bytes, (output)->size, type, low, high) == false) {
                REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: helper_buffer_randint");
                array_dec_refcount(output);
                return false;
            }
            break;

        default:
        printf("In default\n");
        break;
    }


    (*ctx)->output = output;
    return true;

    
    
}

static bool configure_operation_reduction(OperationContext **ctx, OpInstr *instr, ArrayList *list) {
    //WRITE("1.c\n")
    if (!instr || !list) { printf("c0\n"); return false;}

    *ctx = calloc(1, sizeof(OperationContext));
    if ((*ctx) == NULL) { printf("c1\n"); return false; }

    (*ctx)->instr = instr;
    (*ctx)->inputs = list;
    
  //  WRITE("2.c\n")

    dtype common_type;
    Backend common_backend;
    kernel_fn fn;
    kernel_id id;
    IdentityType id_type;
    Iter *iter;
    array *output;

    uint64_t reduction_ndim = instr->reduction.number_reduction_dimensions;
    uint64_t input_ndim = instr->reduction.input_ndim;
    uint64_t *reduction_dims = instr->reduction.reduction_dimensions;
    uint64_t *input_shape = instr->reduction.input_shape;
    bool keepdims = instr->reduction.keepdim;

    bool *reduce_mask = calloc(input_ndim, sizeof(bool));

    for (uint64_t i = 0; i < reduction_ndim; i++) {
        int64_t d = reduction_dims[i];
        if (d < 0) d += input_ndim;
        reduce_mask[d] = true;
    }

  //  WRITE("3.c\n")


    uint64_t *permutate = malloc(sizeof(uint64_t) * input_ndim);

    uint64_t p = 0;

    for (uint64_t i = 0; i < input_ndim; i++) {
        if (!reduce_mask[i]) permutate[p++] = i;
    }

    for (uint64_t i = 0; i < input_ndim; i++) {
        if (reduce_mask[i]) permutate[p++] = i;
    }

 //   WRITE("4.c\n")

    uint64_t num_reduced = 0;

    for (uint64_t i = 0; i < input_ndim; i++) {
        if (reduce_mask[i]) num_reduced++;
    }


    uint64_t *out_shape = malloc(sizeof(uint64_t) * input_ndim);
    uint64_t out_ndim = 0;

    for (uint64_t i = 0; i < input_ndim; i++) {
        if (!reduce_mask[i]) {
            out_shape[out_ndim++] = input_shape[i];
        } else if (keepdims) {
            out_shape[out_ndim++] = 1;
        }
    }

 //   WRITE("5.c\n")


    if (arraylist_cast_to_common_dtype(list) == false) {
       printf("b2\n");
        free(ctx);
        return false;
    }

 //   WRITE("6.c\n")


    if (arraylist_get_common_dtype(list, &common_type) == false) {
       // WRITE("Failed to get common datatype in configure_backwards\n")
        free(ctx);
        return false;
    }

   // WRITE("7.c\n")


    //  printf("Common dtype: %d\n", common_type);

    if (arraylist_get_common_backend(list, &common_backend) == false) {
        //WRITE("Failed to get common backend in configure_backwards\n")
        free(ctx);
        return false;
    }


    if (identity_get_identity_for_op(instr->op, &id_type) == false) {
        printf("cc6\n");
        free(ctx);
        return false;
    }

  //  WRITE("8.c\n")



    if (array_create_array(&output, common_type, out_ndim, out_shape, common_backend) == false) {
        printf("c7\n");
        free(ctx);
        return false;
    }

  //  WRITE("9.c\n")


    if (array_add_buffer(output) == false) {
        printf("c8\n");
        free(ctx);
        array_dec_refcount(output);
        return false;
    }


    if (helper_buffer_with_identity((void **)&output->data, common_type, output->size, id_type) == false) {
        printf("Failed to initialize with identity element\n");
        free(ctx);
        array_dec_refcount(output);
        return false;
    }


    if (arraylist_append(list, output) == false) {
        printf("c9\n");
        free(ctx);
        array_dec_refcount(output);
        return false;
    }


    if (op_type_to_kernel_id(instr->op, &id) == false) {
      //  WRITE("Failed to get kernel if in configure_backwards\n")
        free(ctx);
        return false;
    }

//    WRITE("12.c\n")


    //    WRITE("DEBUG: In configure backwards")
    //   print_kernel_name(id);

    if (dispatch_get_kernel(&fn, common_type, common_backend, id) == false) {
      //  WRITE("Failed to get kernel in configure_backwards\n")
        free(ctx);
        return false;
    }

 //   WRITE("13.c\n")


    IterCtx ctxi = {
        .input_ndim = input_ndim,
        .input_shape = input_shape,
        .num_reduced = num_reduced,
        .reduction_ndims = reduction_ndim,
        .reduction_perm = permutate
    };

   // WRITE("14.c\n")

    if (!Iter_create_iterator(&iter, list, &ctxi, ITER_REDUCTION)) {
        printf("c12\n");
        free(ctx);
        array_dec_refcount(output);
        return false;
    }

 //   WRITE("15.c\n")


    (*ctx)->output = output;
    (*ctx)->iter = iter;
    (*ctx)->fn = fn;
    return true;


}

static bool configure_operation_with_blas(OperationContext **ctx, OpInstr *instr, ArrayList *list) {
    if (!instr || !list) { 
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Opinstr or ArrayList input is not valid");
        return false;
    }

    if (instr->op != OP_MATMUL) {
        REPORT_ERROR(ERR_WRONG_INPUT_FOR_FUNCTION, "configure_operation_with_blas handles matrix & vector operations, check instr->op");
    }


    *ctx = calloc(1, sizeof(OperationContext));
    if ((*ctx) == NULL) {  return false; }

    (*ctx)->instr = instr;
    (*ctx)->inputs = list;

    array *a = list->arrays[0];
    array *b = list->arrays[1];

    uint64_t a_m, a_k;
    uint64_t b_k, b_n;

    if (a->ndim == 1) {
        a_m = 1;
        a_k = a->shape[0];
    } else {
        a_m = a->shape[a->ndim - 2];
        a_k = a->shape[a->ndim - 1];
    }

    if (b->ndim == 1) {
        b_k = b->shape[0];
        b_n = 1;
    } else {
        b_k = b->shape[b->ndim - 2];
        b_n = b->shape[b->ndim - 1];
    }

    if (a_k != b_k) {
        REPORT_ERROR(ERR_SHAPE_MISMATCH, "matmul K mismatch");
        return false;
    }

    IterCtx i_ctx = {
        .input_shape = NULL,
        .reduction_perm = NULL,
        .arr_a_row_mm = a_m,
        .arr_a_col_mm = a_k,
        .arr_b_row_mm = b_k,
        .arr_b_col_mm = b_n
    };

    dtype common_type;
    Backend common_backend;
    uint64_t *shape;
    uint64_t ndim;
    kernel_fn fn;
    kernel_id id;
    Iter *iter;
    array *output;
    
    if (arraylist_get_common_dtype(list, &common_type) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "arraylist_get_common_dtype");
        free_struct(ctx, STRUCT_ITER_CTX);
        return false;
    }

    if (arraylist_get_common_backend(list, &common_backend) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "arraylist_get_common_backend");
        free_struct(ctx, STRUCT_ITER_CTX);
        return false;
    }

    // if (list_to_backend(common_backend, list) == false) {
    //     REPORT_ERROR(ERR_FUNC_CALL_FAILED, "list_to_backend");
    //     free_struct(ctx, STRUCT_ITER_CTX);
    //     return false;
    // }

    if (arraylist_broadcast_for_matmul(list, &shape, &ndim) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "arraylist_broadcast_and_max_ndim");
        free_struct(ctx, STRUCT_ITER_CTX);
        return false;
    }


    if (array_create_array(&output, common_type, ndim, shape, common_backend) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_create_array");
        free_struct(ctx, STRUCT_ITER_CTX);
        return false;
    }

    if (array_add_buffer(output) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_add_buffer");
        free_struct(ctx, STRUCT_ITER_CTX);
        free_struct(output, STRUCT_ARRAY);
        return false;
    }

    if (arraylist_append(list, output) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "arraylist_append");
        free_struct(ctx, STRUCT_ITER_CTX);
        free_struct(output, STRUCT_ARRAY);
        return false;
    }

    if (op_type_to_kernel_id(instr->op, &id) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "op_type_to_kernel_id");
        free_struct(ctx, STRUCT_ITER_CTX);
        free_struct(output, STRUCT_ARRAY);
        return false;
    }


    if (dispatch_get_kernel(&fn, common_type, common_backend, id) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "dispatch_get_kernel");
        free_struct(ctx, STRUCT_ITER_CTX);
        free_struct(output, STRUCT_ARRAY);
        return false;
    }

    if (!Iter_create_iterator(&iter, list, &i_ctx, ITER_BATCHED_MATMUL)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "Iter_create_iterator");
        free_struct(ctx, STRUCT_ITER_CTX);
        free_struct(output, STRUCT_ARRAY);
        return false;
    }

    (*ctx)->output = output;
    (*ctx)->iter = iter;
    (*ctx)->fn = fn;
    return true;

}

static bool configure_backwards(OperationContext **ctx, OpInstr *instr, ArrayList *list) {
    if (!instr || !list) { return false;}

    *ctx = calloc(1, sizeof(OperationContext));
    if ((*ctx) == NULL) { printf("b1\n"); return false; }

    (*ctx)->instr = instr;
    (*ctx)->inputs = list;

    dtype common_type;
    Backend common_backend;
    kernel_fn fn;
    kernel_id id;
    Iter *iter;


    if (arraylist_cast_to_common_dtype(list) == false) {
       // printf("b2\n");
        // free(ctx);
        // return false;
    }


    if (arraylist_get_common_dtype(list, &common_type) == false) {
    //    WRITE("Failed to get common datatype in configure_backwards\n")
        free(ctx);
        return false;
    }

  //  printf("Common dtype: %d\n", common_type);

    if (arraylist_get_common_backend(list, &common_backend) == false) {
    //    WRITE("Failed to get common backend in configure_backwards\n")
        free(ctx);
        return false;
    }


    if (op_type_to_kernel_id(instr->op, &id) == false) {
    //    WRITE("Failed to get kernel if in configure_backwards\n")
        free(ctx);
        return false;
    }

//    WRITE("DEBUG: In configure backwards")
 //   print_kernel_name(id);

    if (dispatch_get_kernel(&fn, common_type, common_backend, id) == false) {
    //    WRITE("Failed to get kernel in configure_backwards\n")
        free(ctx);
        return false;
    }

   // printf("Kernel ID %d\n", id);


    if (!Iter_create_iterator(&iter, list, NULL, ITER_ELEMENTWISE)) {
    //    WRITE("Failed to get create iterator in configure_backwards\n")
        free(ctx);
        return false;
    }


    //(*ctx)->output = NULL;
    (*ctx)->iter = iter;
    (*ctx)->fn = fn;
    //(*ctx)->fn = f64_kernel_add_backward;
    
//    WRITE("End of configure_backwards, returning true\n")
    return true;

 
}

bool configure_operation(OperationContext **ctx, OpInstr *instr, ArrayList *list) {
    
    if (instr->op == OP_MATMUL) {
        if (configure_operation_with_blas(ctx, instr, list) == false) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation_with_blas");
            return false;
        }
        return true;
    }

    if (instr->backward_pass) {
        if (configure_backwards(ctx, instr, list) == false) {
            return false;
        }
        return true;
    } 

    
    if (instr->reduction_op) {
        if (configure_operation_reduction(ctx, instr, list) == false) {
            return false;  
        }
        return true;
    } 

    if (instr->creation_op) {
        if (configure_operation_creation(ctx, instr, list) == false) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation_creation");
            return false;  
        }
        return true;
    }

    if (configure_operation_elementwise(ctx, instr, list) == false) {
        return false;
    }

    return true;
} 


bool op_instr_free(OpInstr *instr) {
    if (!instr) { return false; }

    // if (instr->reduction_dims) {
    //     free(instr->reduction_dims);
    // }

    free(instr);

    return true;
}

bool operation_context_free(OperationContext *ctx, bool free_Opinstr) {
    if (!ctx) { return false; }

    // if (ctx->inputs) {
    //    if (arraylist_free(ctx->inputs) == false) {
    //         printf("F1\n");
    //         return false;
    //    } 
    // }

    // if (ctx->iter) {
    //     if (Iter_free(ctx->iter) == false) {
    //         printf("F2\n");
    //         return false;
    //     } 
        
    // }

    // if (free_Opinstr) {
        
    //     if (ctx->instr) {
    //         if (op_instr_free(ctx->instr) == false) {
    //             printf("F3\n");
    //             return false;
    //         }   
    //     }

    // }

    free(ctx);

    return true;
}

