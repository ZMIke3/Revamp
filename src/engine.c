#include "Engine.h"
#include "Functions.h"

bool value_create_value(Value **out_value, array *data, bool requires_grad, bool is_leaf) {
    if (!out_value) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Value ** is not a valid ptr");
        return false;
    }

    *out_value = malloc(sizeof(Value));
    if (!*out_value) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "Failed to allocate memory of size %zu for Value object", sizeof(Value));
        return false;
    }
    
    if (!data) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "data input is not valid");
        free_struct(*out_value, STRUCT_VALUE);
        return false;
    }

    (*out_value)->data = data;    
    (*out_value)->grad = NULL;     
    (*out_value)->refcount = 1;   
    (*out_value)->can_take_grad = false;  
    (*out_value)->requires_grad = requires_grad;
    (*out_value)->is_leaf = is_leaf;
    (*out_value)->retain_grad = false;
    (*out_value)->creator = NULL;


    if ((*out_value)->requires_grad) {

        array *grad;

        if (array_create_array(&grad, data->type_obj->type, data->ndim, data->shape, data->backend) == false) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: array_create_array");
            free_struct(*out_value, STRUCT_VALUE);
            return false;
        }

        if (helper_buffer_of_zeros((void**)&grad->data, grad->elesize_in_bytes, grad->size) == false) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: helper_buffer_of_zeros");
            free_struct(*out_value, STRUCT_VALUE);
            return false;
        }        

        (*out_value)->grad = grad;

    }

    return true;
}

bool value_set_creator(Value *out_value, Function *fn) {
    if (!out_value || !fn) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Value object or fn object is not valid");
        return false;
    }

    if (function_inc_refcount(fn) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: function_inc_refcount");
        return false;
    }
    out_value->creator = fn;
    out_value->is_leaf = false;

    fn->output = out_value;

    return true;
}

bool value_retain_grad(Value *value) {
    if (!value) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Value object is not valid");
        return false;
    }

    if (value->grad == NULL) {

        array *grad;

        if (array_create_array(&grad, value->data->type_obj->type, value->data->ndim, value->data->shape, value->data->backend) == false) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: array_create_array");
            return false;
        }

        if (helper_buffer_of_zeros((void**)&grad->data, grad->elesize_in_bytes, grad->size) == false) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: helper_buffer_of_zeros");
            free_struct(grad, STRUCT_ARRAY);
            return false;
        }        

        value->grad = grad;

    }

    value->retain_grad = true;
    value->requires_grad = true;
    
    return true;
}

bool value_can_take_grad(Value *value) {
    if (!value) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Value object is not valid");
        return false;
    }

    switch (value->data->type_obj->type) {
        case DTYPE_F32: return true;
        case DTYPE_F64: return true;
        break;
        default: return false;
        break;
    }

    return false;
}

bool value_inc_refcount(Value *value) {
    if (!value) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Value object is not valid");
        return false; 
    }
    value->refcount++;
    return true;
}

bool value_dec_refcount(Value *value) {
    if (!value) { 
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Value object is not valid");
        return false; 
    }

    value->refcount--;
    if (value->refcount == 0) {

        if (value->grad) {

            if (array_dec_refcount(value->grad) == false) {
                REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: array_dec_refcount");
                return false;
            } 
        }

        if (value->data) {

            if (array_dec_refcount(value->data) == false) {
                REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: array_dec_refcount");
                return false;
            }
        }


        if (value->creator) {
            if (function_dec_refcount(value->creator) == false) {
                REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: function_dec_refcount");
                return false;
            }
        }
        
        free_struct(value, STRUCT_VALUE);
    }
    return true;
}



bool value_create(Value **out_value, void *data, dtype type, uint64_t ndim, uint64_t *shape, Backend backend) {
  
  array *arr = NULL;

  if (array_create_array(&arr, type, ndim, shape, backend) == false) {
      REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_create_array");
      return false;
  }

  if (array_add_data(arr, data) == false) {
      REPORT_ERROR(ERR_FUNC_CALL_FAILED, "array_add_data");
      free_struct(arr, STRUCT_ARRAY);
      return false;
  }

  if (value_create_value(out_value, arr, true, true) == false) {
      REPORT_ERROR(ERR_FUNC_CALL_FAILED, "value_create_value");
      free_struct(arr, STRUCT_ARRAY);
      return false;
  }

  return true;


}

bool value_get_array(Value *v, array **out_array) {
    if (!v || !out_array) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Value object is not valid or out_array input is not a valid ptr");
        return false;
    }
    if (!v->data) {
        REPORT_ERROR(ERR_STRUCT_FIELD_NOT_INIT, "Value struct field data is not initialized");
        return false;
    }
    *out_array = v->data;
    return true;
}

bool value_get_shape(Value *v, uint64_t **out_shape) {
    if (!v || !out_shape) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Value object is not valid or out_shape input is not a valid ptr");
        return false;
    }
    if (!v->data) {
        REPORT_ERROR(ERR_STRUCT_FIELD_NOT_INIT, "Value struct field 'data' is not initialized");
        return false;
    }
    return array_get_shape(v->data, out_shape);
}

bool value_get_stride(Value *v, int64_t **out_stride) {
    if (!v || !out_stride) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Value object is not valid or out_stride input is not a valid ptr");
        return false;
    }
    if (!v->data) {
        REPORT_ERROR(ERR_STRUCT_FIELD_NOT_INIT, "Value struct field 'data' is not initialized");
        return false;
    }
    return array_get_stride(v->data, out_stride);
}

bool value_get_ndim(Value *v, uint64_t *out_ndim) {
    if (!v || !out_ndim) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Value object is not valid or out_ndim input is not a valid ptr");
        return false;
    }
    if (!v->data) {
        REPORT_ERROR(ERR_STRUCT_FIELD_NOT_INIT, "Value struct field data is not initialized");
        return false;
    }
    return array_get_ndim(v->data, out_ndim);
}

bool value_get_size(Value *v, uint64_t *out_size) {
    if (!v || !out_size) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Value object is not valid or out_size input is not a valid ptr");
        return false;
    }
    if (!v->data) {
        REPORT_ERROR(ERR_STRUCT_FIELD_NOT_INIT, "Value struct field data is not initialized");
        return false;
    }
    return array_get_size(v->data, out_size);
}

bool value_get_elesize(Value *v, int *out_elesize) {
    if (!v || !out_elesize) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Value object is not valid or out_elesize input is not a valid ptr");
        return false;
    }
    if (!v->data) {
        REPORT_ERROR(ERR_STRUCT_FIELD_NOT_INIT, "Value struct field data is not initialized");
        return false;
    }
    return array_get_elesize(v->data, out_elesize);
}

bool value_get_dtype(Value *v, dtype *out_dtype) {
    if (!v || !out_dtype) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Value object is not valid or out_dtype input is not a valid ptr");
        return false;
    }
    if (!v->data) {
        REPORT_ERROR(ERR_STRUCT_FIELD_NOT_INIT, "Value struct field data is not initialized");
        return false;
    }
    return array_get_dtype(v->data, out_dtype);
}

bool value_get_backend(Value *v, Backend *out_backend) {
    if (!v || !out_backend) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Value object is not valid or out_backend input is not a valid ptr");
        return false;
    }
    if (!v->data) {
        REPORT_ERROR(ERR_STRUCT_FIELD_NOT_INIT, "Value struct field data is not initialized");
        return false;
    }
    return array_get_backend(v->data, out_backend);
}

bool value_print_value(Value *v, bool print_grad) {
    if (!v) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Value object is not valid in value_print_value");
        return false;
    }

    if (print_grad) {
        if (!v->grad) {
            REPORT_ERROR(ERR_STRUCT_FIELD_NOT_INIT, "Value struct field 'grad' is not initialized in value_print_value");
            return false;
        }
        array_print_array(v->grad);
    } else {
        if (!v->data) {
            REPORT_ERROR(ERR_STRUCT_FIELD_NOT_INIT, "Value struct field 'data' is not initialized in value_print_value");
            return false;
        }
        array_print_array(v->data);
    }

    return true;
}


bool function_create_function(Function **out_fn, bool (*backward_fn)(struct Function*)) {
    if (!out_fn || !backward_fn) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Function **object or backward_fn are not valid");
        return false;
    }

    Function *fn = malloc(sizeof(Function));
    if (!fn) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "Failed to allocate memory of size %zu for function object", malloc(sizeof(Function)));
        return false;
    }

    fn->capacity = 2; 
    fn->inputs = calloc(fn->capacity, sizeof(Value*));
    if (!fn->inputs) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "Failed to allocate memory of size %zu for function inputs", fn->capacity*sizeof(Value*));
        free_struct(fn, STRUCT_FUNCTION);
        return false;
    }
    fn->backward = backward_fn;
    fn->refcount = 1; 
    fn->num_inputs = 0;
    fn->visited = false;
    fn->ctx = NULL;
    fn->output = NULL;

    *out_fn = fn;
    return true;
}

bool function_append_input(Function *fn, Value *input) {
    if (!fn || !input) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Function object or Value object is not valid");
        return false;
    }

    if (fn->num_inputs >= fn->capacity) { 
        REPORT_ERROR(ERR_STRUCT_FIELD_OUT_OF_BOUNDS, "function object num_inputs %d must be < than capacity %d", fn->num_inputs, fn->capacity);
        return false; 
    }

    if (fn->num_inputs >= fn->capacity) {
        int new_capacity = fn->capacity * 2;
        Value **new_inputs = (Value**)realloc(fn->inputs, new_capacity * sizeof(Value*));
        
        if (!new_inputs) {
            REPORT_ERROR(ERR_MALLOC_FAILED, "Failed to reallocate memory of size %zu for Value objects", new_capacity * sizeof(Value*));
            return false;
        }
        for (int i = fn->capacity; i < new_capacity; i++) new_inputs[i] = NULL;

        fn->inputs = new_inputs;
        fn->capacity = new_capacity;
    }


    if (value_inc_refcount(input) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: value_inc_refcount");
        return false;
    }

    fn->inputs[fn->num_inputs] = input;
    fn->num_inputs++;
    return true;
}

bool function_inc_refcount(Function *fn) {
    if (!fn) { 
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Function object is not valid");
        return false; 
    }
    fn->refcount++;
    return true;
}

bool function_dec_refcount(Function *fn) {
    if (!fn) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Function object is not valid");
        return false;
    }

    fn->refcount--;
    if (fn->refcount == 0) {
        for (int i = 0; i < fn->num_inputs; i++) {
            if (fn->inputs[i]) {
                if (value_dec_refcount(fn->inputs[i]) == false) {
                    REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: value_dec_refcount");
                    return false;
                }
            }
        }
        
        free_struct(fn, STRUCT_FUNCTION);
    }
    return true;
}



static bool __function_set_visited__(Function *fn, bool visited) {
    if (!fn) { 
       REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Function object is not vlaid");
       return false; 
    }
    fn->visited = visited;
    return true;
}

static bool __build_topo__(Function *fn, List_FunctionPtr *list_fn, int v) {
    if (!fn || !list_fn) { 
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Function object or list_fnPtr object is not valid");
        return false; 
    }
    
    if (fn->visited) {
        return true;
    }

    fn->visited = true;

    for (int i = 0; i < fn->num_inputs; i++) {
        Value *value = fn->inputs[i];
        if (value && value->creator) {
            if (__build_topo__(value->creator, list_fn, v + 1) == false) {
                REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: __build_topo__");
                return false;
            }
        }
    }


    if (list_FunctionPtr_push(list_fn, fn) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: list_FunctionPtr_push");
        return false;
    }

    return true;
}

static bool __free_grad_if_not_leaf__(List_FunctionPtr *list_fn) {
    if (!list_fn) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "List_FnPtr object is not valid");
        return false;
    }

    for (int i = 0; i < list_fn->size - 1; i++) {
        Function *fn;

        if (list_FunctionPtr_get(list_fn, i, &fn) == false) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: list_FunctionPtr_get");
            return false;
        }

        if (!fn) {
            REPORT_ERROR(ERR_STRUCT_NOT_INIT, "fn ptr at idx %d from list is not valid", i);
            return false;
        }

        if (!fn->output->grad) {
            REPORT_ERROR(ERR_STRUCT_FIELD_NOT_INIT, "function output grad is not initialized");
            return false;
        }

        if (!fn->output->is_leaf && !fn->output->retain_grad) {
            if (array_dec_refcount(fn->output->grad) == false) {
                REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: array_dec_refcount");
            }
            fn->output->grad = NULL;
        }
    }

    return true;
}

bool backward(Value *value) {
    if (!value) { 
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Value object is not valid");
        return false; 
    }

    if (value->requires_grad == false) {
        REPORT_ERROR(ERR_INPUT_OUT_OF_BOUNDS, "Called backward for Value object that does not require grad");
        return false;
    }

    if (!value->grad) { 
        REPORT_ERROR(ERR_STRUCT_FIELD_NOT_INIT, "Value grad field not initialized");
        return false; 
    }

    if (!value->creator) {
        REPORT_ERROR(ERR_STRUCT_FIELD_NOT_INIT, "Value creator field not initialized");
        return false; 
    }

    List_FunctionPtr *list_fn;

    if (list_FunctionPtr_create(&list_fn, 10) == false) {
        return false;
    }

    if (__build_topo__(value->creator, list_fn, 0) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: __build_topo__");
        return false;
    }

    if (helper_buffer_of_ones((void**)&value->grad->data, value->grad->elesize_in_bytes, value->grad->size, value->grad->type_obj->type) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: helper_buffer_of_ones");
        return false;
    }    
    
    for (int i = list_fn->size - 1; i >= 0; i--) {
        Function *fn;

        if (list_FunctionPtr_get(list_fn, i, &fn) == false) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: list_FunctionPtr_get");
            return false;
        }

        if (!fn->backward) {
            REPORT_ERROR(ERR_STRUCT_FIELD_NOT_INIT, "function backward field not initialized");
            return false;
        }

         fn->backward(fn);
    }

    for (int i = 0; i < list_fn->size; i++) {

        Function *fn;

        if (list_FunctionPtr_get(list_fn, i, &fn) == false) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: list_FunctionPtr_get");
            return false;
        }

        if (__function_set_visited__(fn, false) == false) {
            REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: __function_set_visited__");
            return false;
        }
    }
    

    if (list_FunctionPtr_free(list_fn) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: list_FunctionPtr_free");
        return false;
    }

    return true;

}

