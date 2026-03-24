#ifndef ENGINE_H
#define ENGINE_H

#include "array.h"
#include "helpers.h"
#include "mem.h"
#include "dtypes.h"
#include "backend.h"

typedef struct array array;

typedef struct Value {
    array *data;
    array *grad;
    uint64_t refcount;
    uint16_t v_id;
    bool requires_grad;
    bool is_leaf;
    bool can_take_grad;
    bool retain_grad;
    struct Function *creator;
} Value;

typedef struct Function {
    bool (*backward)(struct Function *self);
    Value **inputs;
    Value *output;
    void *ctx;
    uint64_t refcount;
    int num_inputs;
    int capacity;
    bool visited;
    const char *name;
    uint16_t f_id;
} Function;

// typedef Function* FunctionPtr;
// DEFINE_LIST(FunctionPtr)



bool value_create_value(Value **out_value, array *data, bool requires_grad, bool is_leaf);
bool value_set_creator(Value *out_value, Function *fn);
bool value_retain_grad(Value *value);
bool value_can_take_grad(Value *value);

bool value_create(Value **out_value, void *data, dtype type, uint64_t ndim, uint64_t *shape, Backend backend);
bool value_get_array(Value *v, array **out_array);
bool value_get_shape(Value *v, uint64_t **out_shape);
bool value_get_stride(Value *v, int64_t **out_stride);
bool value_get_ndim(Value *v, uint64_t *out_ndim);
bool value_get_size(Value *v, uint64_t *out_size);
bool value_get_elesize(Value *v, int *out_elesize);
bool value_get_dtype(Value *v, dtype *out_dtype);
bool value_get_backend(Value *v, Backend *out_backend);
bool value_print_value(Value *v, bool print_grad);


bool value_inc_refcount(Value *value);
bool value_dec_refcount(Value *value);

bool function_create_function(Function **out_fn, bool (*backward_fn)(struct Function*));
bool function_append_input(Function *fn, Value *input);

bool function_inc_refcount(Function *fn);
bool function_dec_refcount(Function *fn);

bool engine_new_id(uint16_t *id);

bool backward(Value *value);



#endif // ENGINE_H