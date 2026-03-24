#ifndef MEM_H
#define MEM_H

typedef enum {
    STRUCT_ARRAY,
    STRUCT_VALUE,
    STRUCT_FUNCTION,
    STRUCT_ITER,
    STRUCT_ITER_CTX,
    STRUCT_KERNEL_INFO,
    STRUCT_OP_INSTR,
    STRUCT_OPERATION_CONTEXT,
} StructType;

// Forward declarations - no headers needed for these
typedef struct OperationContext OperationContext; 
typedef struct OpInstr OpInstr;
typedef struct Value Value;
typedef struct array array;
typedef struct Function Function;

void free_struct(void *s, StructType type);

#endif // MEM_H