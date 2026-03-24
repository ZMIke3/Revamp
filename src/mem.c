#include "mem.h"
#include "array.h"
#include "Engine.h"
#include "Iter.h"
#include "Ops.h"

void free_struct(void *s, StructType type) {

    if (!s) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "");
        return;
    }

    switch (type) {

        case STRUCT_ARRAY: {
            array *a = (array *)s;
            if (a->owns_data) free(a->data);
            free(a->shape);
            free(a->stride);
            free(a->type_obj);
            free(a);
            break;
        }

        case STRUCT_VALUE: {
            Value *v = (Value *)s;
            if (v->data) free_struct(v->data, STRUCT_ARRAY);
            if (v->grad) free_struct(v->grad, STRUCT_ARRAY);
            free(v);
            break;
        }

        case STRUCT_FUNCTION: {
            Function *fn = (Function *)s;
            free(fn->inputs);
            free(fn->ctx);
            free(fn);
            break;
        }

        // case STRUCT_ITER: {
        //     Iter *it = (Iter *)s;
        //     int nops = it->nops;

        //     for (int i = 0; i < nops; i++) {
        //         free(it->strides[i]);
        //         free(it->backstrides[i]);
        //     }
        //     free(it->strides);
        //     free(it->backstrides);

        //     for (int i = 0; i < nops; i++) {
        //         free(it->reduce_strides[i]);
        //     }
        //     free(it->reduce_strides);

        //     free(it->baseptrs);
        //     free(it->dataptrs);
        //     free(it->shape);
        //     free(it->coordinates);
        //     free(it->inner_strides);
        //     free(it->_reduce_dims);
        //     free(it->reduce_shape);
        //     free(it->flat_strides);
        //     free(it);
        //     break;
        // }

        case STRUCT_ITER_CTX: {
            IterCtx *ic = (IterCtx *)s;
            free(ic->input_shape);
            free(ic->reduction_perm);
            free(ic);
            break;
        }

        case STRUCT_KERNEL_INFO: {
            kernel_info *ki = (kernel_info *)s;
            free(ki->dataptrs);
            free(ki->saved_dataptrs);
            free(ki->inner_strides);
            free(ki->saved_dataptrs_inner_strides);
            free(ki->reduce_shape);
            free(ki->flat_strides);

            for (uint64_t i = 0; i < ki->rndims; i++) {
                free(ki->reduce_strides[i]);
            }
            free(ki->reduce_strides);
            free(ki);
            break;
        }

        case STRUCT_OP_INSTR: {
            OpInstr *oi = (OpInstr *)s;
            free(oi->creation.shape);
            free(oi->reduction.input_shape);
            free(oi->reduction.reduction_dimensions);
            free(oi);
            break;
        }

        case STRUCT_OPERATION_CONTEXT: {
            OperationContext *oc = (OperationContext *)s;
            if (oc->iter)  free_struct(oc->iter,  STRUCT_ITER);
            if (oc->instr) free_struct(oc->instr, STRUCT_OP_INSTR);
            free(oc);
            break;
        }

        default: {
            REPORT_ERROR(ERR_INPUT_OUT_OF_BOUNDS, "free_struct: unknown StructType %d", type);
            break;
        }
    }
}


