#ifndef OPS_H
#define OPS_H

#include "array.h"
#include "dtypes.h"
#include "Dispatch.h"
#include "arraylist.h"
#include <stdint.h>
#include <float.h>  
#include <limits.h> 
#include "Iter.h"

typedef enum {
    OP_ADD = 0, 
    OP_SUBTRACT,
    OP_MULTIPLY,
    OP_DIVIDE,
    OP_MATMUL,

    OP_SQUARE_ROOT,
    OP_ABSOLUTE_VALUE,
    OP_NEGATION,
    OP_POWER,
    OP_LOG,
    OP_EXP,

    OP_EQUALITY,
    OP_GREATER_THAN,
    OP_LESS_THAN,
    OP_ELEMENT_WISE_MAXIMUM,
    OP_ELEMENT_WISE_MINIMUM,

    
    OP_ZEROS,
    OP_ONES,
    OP_ZEROS_LIKE,
    OP_ONES_LIKE,
    OP_ARANGE,
    OP_LINSPACE,
    OP_RANDU,
    OP_RANDN,
    OP_RANDINT,
    OP_FILL,

    
    OP_SUM_REDUCTION,
    OP_MAX_REDUCTION,
    OP_MIN_REDUCTION,
    OP_MEAN_REDUCTION,
    OP_ARGMAX_REDUCTION,
    OP_ARGMIN_REDUCTION,
    OP_WHERE,


    OP_ADD_BACKWARDS,
    OP_SUBTRACT_BACKWARDS,
    OP_MULTIPLY_BACKWARDS,
    OP_DIVIDE_BACKWARDS,
    OP_POWER_BACKWARDS,
    OP_LOG_BACKWARDS,
    OP_EXP_BACKWARDS, 
}OpType;

typedef enum {
    IDENTITY_ZERO,
    IDENTITY_ONE,
    IDENTITY_MIN, 
    IDENTITY_MAX, 
} IdentityType;

typedef struct OpInstr {
    union {
        
        struct {
            uint64_t *shape;
            uint64_t  ndim;
            dtype dtype;
            Backend backend;
            double fill_value;
            double start;
            double step;
            double stop;
            double low;
            double high;
            double mean;
            double stdev;
        }creation;

        struct {
            int64_t number_reduction_dimensions;
            uint64_t *reduction_dimensions;
            uint64_t *input_shape;
            uint64_t input_ndim;
            bool keepdim;
        }reduction;       
         
    };
    
    int inArity;
    int outArity;
    bool reduction_op;
    bool creation_op;
    bool backward_pass;
    bool force_cast_inputs;
    dtype cast_to;
    OpType op;

}OpInstr;

typedef struct OperationContext{
    OpInstr *instr;
    ArrayList *inputs;
    Iter *iter;
    kernel_fn fn;
    array *output;
}OperationContext;


bool configure_operation(OperationContext **ctx, OpInstr *instr, ArrayList *list);
bool op_instr_free(OpInstr *instr);
bool operation_context_free(OperationContext *ctx, bool free_Opinstr);

#endif // OPS_H