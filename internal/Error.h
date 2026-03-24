#ifndef ERROR_H
#define ERROR_H

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

#define ERROR_TABLE(X) \
    X(ERR_MALLOC_FAILED,  "Memory allocation failed") \
    X(ERR_SHAPE_MISMATCH, "Incompatible array shapes") \
    X(ERR_DTYPE_MISMATCH, "Unsupported data type for operation") \
    X(ERR_FUNC_CALL_FAILED, "Function call returned false") \
    X(ERR_INPUT_IS_NOT_INIT, "Input to function is not initialized") \
    X(ERR_STRUCT_FIELD_NOT_INIT, "Struct field is not initialized") \
    X(ERR_FAILED_TO_CREATE_ARRAY, "Failed to create array object") \
    X(ERR_INPUT_OUT_OF_BOUNDS, "Input is out of bounds") \
    X(ERR_STRUCT_FIELD_OUT_OF_BOUNDS, "Struct field is illogical") \
    X(ERR_STRUCT_NOT_INIT, "Struct is not initialized") \
    X(ERR_WRONG_INPUT_FOR_FUNCTION, "Input is not valid for the function") \
    X(NOT_IMPLEMENTED, "Function is not implemented")\



typedef enum {
#define X(id, msg) id,
    ERROR_TABLE(X)
#undef X
} error_id;

#define MAX_STACK_ERROR 64

typedef struct error{
    int code;           
    const char* file;   
    int line;           
    char custom_msg[256];
} error;


typedef struct error_ctx{
    error stack[MAX_STACK_ERROR];
    int top;
}error_ctx;

extern error_ctx err_t;


#define REPORT_ERROR(id, msg, ...) do { \
    if (err_t.top < MAX_STACK_ERROR - 1) { \
        err_t.top++; \
        error *e = &err_t.stack[err_t.top]; \
        e->code = id; \
        e->file = __FILE__; \
        e->line = __LINE__; \
        snprintf(e->custom_msg, 256, msg, ##__VA_ARGS__); \
    } \
} while(0)


const char* error_get_msg(error_id id);

void error_stack_print();

#endif // ERROR_H