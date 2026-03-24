#include "helpers.h"

uint64_t s[2]; 
static bool rng_initialized = false;

static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

void helper_rand_seed(uint64_t seed) {
    uint64_t cpu_time = seed;
    s[0] = splitmix64(&cpu_time);
    s[1] = splitmix64(&cpu_time);
    rng_initialized = true;
}

static inline uint64_t xorshift128plus(void) {
    if (!rng_initialized) {
        uint64_t cpu_time = (uint64_t)time(NULL);
        s[0] = splitmix64(&cpu_time);
        s[1] = splitmix64(&cpu_time);
        rng_initialized = true;
    }
    
    uint64_t x = s[0];
    uint64_t const y = s[1];
    
    s[0] = y;
    x ^= x << 23;
    s[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
    
    return s[1] + y;
}

static inline double rand_uniform() {
    return (double)xorshift128plus() / (double)UINT64_MAX;
}

static inline double rand_uniform_range(double low, double high) {
    return low + rand_uniform() * (high - low);
}

static inline double rand_normal(double mean, double stddev) {
    static bool has_spare = false;
    static double spare;
    
    if (has_spare) {
        has_spare = false;
        return mean + stddev * spare;
    }
    
    has_spare = true;
    double u = rand_uniform();
    double v = rand_uniform();
    double r = sqrt(-2.0 * log(u));
    double theta = 2.0 * PI * v;
    
    spare = r * sin(theta);
    return mean + stddev * r * cos(theta);
}

bool helper_calc_stride(int64_t **out_stride, uint64_t ndim, uint64_t *shape, size_t dtype_size) {

    *out_stride = malloc(ndim * sizeof(int64_t));

    if (out_stride == NULL) {
        return false;
    }

    (*out_stride)[ndim - 1] = dtype_size;

    for (int64_t i = ndim - 2; i >= 0; i--) {
        (*out_stride)[i] = (*out_stride)[i + 1] * shape[i + 1];
    }

    return true;
}

bool helper_calc_backstride(int64_t **out_backstride, uint64_t ndim, int64_t *strides, uint64_t *shape) {

    *out_backstride = malloc(ndim * sizeof(int64_t));

    if (out_backstride == NULL) {
        return false;
    }

    for (int dim = 0; dim < ndim; dim++) {
        (*out_backstride)[dim] = strides[dim] * (shape[dim] - 1);
    }

    return true;
}


bool helper_buffer_of_zeros(void **in_ptr, size_t size_of_element, uint64_t number_ele) {
    *in_ptr = calloc(number_ele, size_of_element);
    if (!*in_ptr) { return false; }
    return true;
}

bool helper_buffer_of_ones(void **in_ptr, size_t size_of_element, uint64_t number_ele, dtype type) {
    if (!in_ptr || size_of_element == 0) {
        return false;
    }

    *in_ptr = malloc(number_ele * size_of_element);
    if (!*in_ptr) {
        return false;
    }

    switch (type) {
        case DTYPE_I8: {
            int8_t *ptr = (int8_t *)(*in_ptr);
            for (uint64_t i = 0; i < number_ele; i++) { ptr[i] = 1; }
            break;
        }
        case DTYPE_I16: {
            int16_t *ptr = (int16_t *)(*in_ptr);
            for (uint64_t i = 0; i < number_ele; i++) { ptr[i] = 1; }
            break;
        }
        case DTYPE_I32: {
            int32_t *ptr = (int32_t *)(*in_ptr);
            for (uint64_t i = 0; i < number_ele; i++) { ptr[i] = 1; }
            break;
        }
        case DTYPE_I64: {
            int64_t *ptr = (int64_t *)(*in_ptr);
            for (uint64_t i = 0; i < number_ele; i++) { ptr[i] = 1; }
            break;
        }
        case DTYPE_U8: {
            uint8_t *ptr = (uint8_t *)(*in_ptr);
            for (uint64_t i = 0; i < number_ele; i++) { ptr[i] = 1; }
            break;
        }
        case DTYPE_U16: {
            uint16_t *ptr = (uint16_t *)(*in_ptr);
            for (uint64_t i = 0; i < number_ele; i++) { ptr[i] = 1; }
            break;
        }
        case DTYPE_U32: {
            uint32_t *ptr = (uint32_t *)(*in_ptr);
            for (uint64_t i = 0; i < number_ele; i++) { ptr[i] = 1; }
            break;
        }
        case DTYPE_U64: {
            uint64_t *ptr = (uint64_t *)(*in_ptr);
            for (uint64_t i = 0; i < number_ele; i++) { ptr[i] = 1; }
            break;
        }
        case DTYPE_F32: {
        //    printf("Yes 0\n");
            float *ptr = (float *)(*in_ptr);
            for (uint64_t i = 0; i < number_ele; i++) { ptr[i] = 1.0f; }
            break;
        }
        case DTYPE_F64: {
         //   printf("Yes 1\n");
            double *ptr = (double *)(*in_ptr);
            for (uint64_t i = 0; i < number_ele; i++) { ptr[i] = 1.0; }
            break;
        }
        case DTYPE_BOOL: {
            bool *ptr = (bool *)(*in_ptr);
            for (uint64_t i = 0; i < number_ele; i++) { ptr[i] = true; }
            break;
        }
        default: {
            free(*in_ptr);
            *in_ptr = NULL;
            return false;
        }
    }

    return true;
}

bool helper_buffer_with_value(void **buffer, size_t elesize, uint64_t size, dtype type, double value) {
    if (!buffer) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "buffer pointer is NULL");
        return false;
    }

    *buffer = malloc(elesize * size);
    if (*buffer == NULL) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "Failed to allocate buffer in helper_buffer_with_value");
        return false;
    }

    switch (type) {
        case DTYPE_F64: {
            double *buf = (double *)*buffer;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = value;
            }
            break;
        }
        case DTYPE_F32: {
            float *buf = (float *)*buffer;
            float val = (float)value;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = val;
            }
            break;
        }
        case DTYPE_I64: {
            int64_t *buf = (int64_t *)*buffer;
            int64_t val = (int64_t)value;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = val;
            }
            break;
        }
        case DTYPE_I32: {
            int32_t *buf = (int32_t *)*buffer;
            int32_t val = (int32_t)value;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = val;
            }
            break;
        }
        case DTYPE_I16: {
            int16_t *buf = (int16_t *)*buffer;
            int16_t val = (int16_t)value;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = val;
            }
            break;
        }
        case DTYPE_I8: {
            int8_t *buf = (int8_t *)*buffer;
            int8_t val = (int8_t)value;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = val;
            }
            break;
        }
        case DTYPE_U64: {
            uint64_t *buf = (uint64_t *)*buffer;
            uint64_t val = (uint64_t)value;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = val;
            }
            break;
        }
        case DTYPE_U32: {
            uint32_t *buf = (uint32_t *)*buffer;
            uint32_t val = (uint32_t)value;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = val;
            }
            break;
        }
        case DTYPE_U16: {
            uint16_t *buf = (uint16_t *)*buffer;
            uint16_t val = (uint16_t)value;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = val;
            }
            break;
        }
        case DTYPE_U8: {
            uint8_t *buf = (uint8_t *)*buffer;
            uint8_t val = (uint8_t)value;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = val;
            }
            break;
        }
        default:
            free(*buffer);
            *buffer = NULL;
            REPORT_ERROR(ERR_DTYPE_MISMATCH, "Unsupported dtype in helper_buffer_with_value");
            return false;
    }

    return true;
}

bool helper_buffer_arange(void **buffer, size_t elesize, uint64_t size, dtype type, double start, double step) {
    if (!buffer) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "buffer pointer is NULL");
        return false;
    }

    *buffer = malloc(elesize * size);
    if (*buffer == NULL) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "Failed to allocate buffer in helper_buffer_arange");
        return false;
    }

    switch (type) {
        case DTYPE_F64: {
            double *buf = (double *)*buffer;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = start + i * step;
            }
            break;
        }
        case DTYPE_F32: {
            float *buf = (float *)*buffer;
            float start_f = (float)start;
            float step_f = (float)step;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = start_f + i * step_f;
            }
            break;
        }
        case DTYPE_I64: {
            int64_t *buf = (int64_t *)*buffer;
            int64_t start_i = (int64_t)start;
            int64_t step_i = (int64_t)step;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = start_i + i * step_i;
            }
            break;
        }
        case DTYPE_I32: {
            int32_t *buf = (int32_t *)*buffer;
            int32_t start_i = (int32_t)start;
            int32_t step_i = (int32_t)step;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = start_i + i * step_i;
            }
            break;
        }
        case DTYPE_I16: {
            int16_t *buf = (int16_t *)*buffer;
            int16_t start_i = (int16_t)start;
            int16_t step_i = (int16_t)step;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = start_i + i * step_i;
            }
            break;
        }
        case DTYPE_I8: {
            int8_t *buf = (int8_t *)*buffer;
            int8_t start_i = (int8_t)start;
            int8_t step_i = (int8_t)step;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = start_i + i * step_i;
            }
            break;
        }
        case DTYPE_U64: {
            uint64_t *buf = (uint64_t *)*buffer;
            uint64_t start_i = (uint64_t)start;
            uint64_t step_i = (uint64_t)step;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = start_i + i * step_i;
            }
            break;
        }
        case DTYPE_U32: {
            uint32_t *buf = (uint32_t *)*buffer;
            uint32_t start_i = (uint32_t)start;
            uint32_t step_i = (uint32_t)step;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = start_i + i * step_i;
            }
            break;
        }
        case DTYPE_U16: {
            uint16_t *buf = (uint16_t *)*buffer;
            uint16_t start_i = (uint16_t)start;
            uint16_t step_i = (uint16_t)step;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = start_i + i * step_i;
            }
            break;
        }
        case DTYPE_U8: {
            uint8_t *buf = (uint8_t *)*buffer;
            uint8_t start_i = (uint8_t)start;
            uint8_t step_i = (uint8_t)step;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = start_i + i * step_i;
            }
            break;
        }
        default:
            free(*buffer);
            *buffer = NULL;
            REPORT_ERROR(ERR_DTYPE_MISMATCH, "Unsupported dtype in helper_buffer_arange");
            return false;
    }

    return true;
}

bool helper_buffer_linspace(void **buffer, size_t elesize, uint64_t size, dtype type, double start, double stop) {
    if (!buffer) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "buffer pointer is NULL");
        return false;
    }

    if (size == 0) {
        REPORT_ERROR(ERR_INPUT_OUT_OF_BOUNDS, "size must be > 0 for linspace");
        return false;
    }

    *buffer = malloc(elesize * size);
    if (*buffer == NULL) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "Failed to allocate buffer in helper_buffer_linspace");
        return false;
    }

    // Calculate step: (stop - start) / (size - 1)
    // Special case: if size == 1, just use start
    double step = (size > 1) ? (stop - start) / (size - 1) : 0.0;

    switch (type) {
        case DTYPE_F64: {
            double *buf = (double *)*buffer;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = start + i * step;
            }
            // Ensure last element is exactly stop to avoid floating point errors
            if (size > 1) buf[size - 1] = stop;
            break;
        }
        case DTYPE_F32: {
            float *buf = (float *)*buffer;
            float start_f = (float)start;
            float stop_f = (float)stop;
            float step_f = (float)step;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = start_f + i * step_f;
            }
            if (size > 1) buf[size - 1] = stop_f;
            break;
        }
        case DTYPE_I64: {
            int64_t *buf = (int64_t *)*buffer;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = (int64_t)(start + i * step);
            }
            break;
        }
        case DTYPE_I32: {
            int32_t *buf = (int32_t *)*buffer;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = (int32_t)(start + i * step);
            }
            break;
        }
        case DTYPE_I16: {
            int16_t *buf = (int16_t *)*buffer;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = (int16_t)(start + i * step);
            }
            break;
        }
        case DTYPE_I8: {
            int8_t *buf = (int8_t *)*buffer;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = (int8_t)(start + i * step);
            }
            break;
        }
        case DTYPE_U64: {
            uint64_t *buf = (uint64_t *)*buffer;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = (uint64_t)(start + i * step);
            }
            break;
        }
        case DTYPE_U32: {
            uint32_t *buf = (uint32_t *)*buffer;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = (uint32_t)(start + i * step);
            }
            break;
        }
        case DTYPE_U16: {
            uint16_t *buf = (uint16_t *)*buffer;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = (uint16_t)(start + i * step);
            }
            break;
        }
        case DTYPE_U8: {
            uint8_t *buf = (uint8_t *)*buffer;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = (uint8_t)(start + i * step);
            }
            break;
        }
        default:
            free(*buffer);
            *buffer = NULL;
            REPORT_ERROR(ERR_DTYPE_MISMATCH, "Unsupported dtype in helper_buffer_linspace");
            return false;
    }

    return true;
}

bool helper_buffer_randu(void **buffer, size_t elesize, uint64_t size, dtype type, double low, double high) {
    if (!buffer) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "buffer pointer is NULL");
        return false;
    }

    *buffer = malloc(elesize * size);
    if (*buffer == NULL) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "Failed to allocate buffer in helper_buffer_rand");
        return false;
    }

    switch (type) {
        case DTYPE_F64: {
            double *buf = (double *)*buffer;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = rand_uniform_range(low, high);
            }
            break;
        }
        case DTYPE_F32: {
            float *buf = (float *)*buffer;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = (float)rand_uniform_range(low, high);
            }
            break;
        }
        default:
            free(*buffer);
            *buffer = NULL;
            REPORT_ERROR(ERR_DTYPE_MISMATCH, "Random generation only supported for float types");
            return false;
    }

    return true;
}

bool helper_buffer_randn(void **buffer, size_t elesize, uint64_t size, dtype type, double mean, double stddev) {
    if (!buffer) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "buffer pointer is NULL");
        return false;
    }

    *buffer = malloc(elesize * size);
    if (*buffer == NULL) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "Failed to allocate buffer in helper_buffer_randn");
        return false;
    }

    switch (type) {
        case DTYPE_F64: {
            double *buf = (double *)*buffer;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = rand_normal(mean, stddev);
            }
            break;
        }
        case DTYPE_F32: {
            float *buf = (float *)*buffer;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = (float)rand_normal(mean, stddev);
            }
            break;
        }
        default:
            free(*buffer);
            *buffer = NULL;
            REPORT_ERROR(ERR_DTYPE_MISMATCH, "Random generation only supported for float types");
            return false;
    }

    return true;
}

bool helper_buffer_randint(void **buffer, size_t elesize, uint64_t size, dtype type, int64_t low, int64_t high) {
    if (!buffer) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "buffer pointer is NULL");
        return false;
    }
    if (low >= high) {
        REPORT_ERROR(ERR_INPUT_OUT_OF_BOUNDS, "low must be < high in randint");
        return false;
    }

    *buffer = malloc(elesize * size);
    if (*buffer == NULL) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "Failed to allocate buffer in helper_buffer_randint");
        return false;
    }

    uint64_t range = (uint64_t)(high - low);

    switch (type) {
        case DTYPE_I64: {
            int64_t *buf = (int64_t *)*buffer;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = low + (int64_t)(xorshift128plus() % range);
            }
            break;
        }
        case DTYPE_I32: {
            int32_t *buf = (int32_t *)*buffer;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = (int32_t)(low + (int64_t)(xorshift128plus() % range));
            }
            break;
        }
        case DTYPE_I16: {
            int16_t *buf = (int16_t *)*buffer;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = (int16_t)(low + (int64_t)(xorshift128plus() % range));
            }
            break;
        }
        case DTYPE_I8: {
            int8_t *buf = (int8_t *)*buffer;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = (int8_t)(low + (int64_t)(xorshift128plus() % range));
            }
            break;
        }
        case DTYPE_U64: {
            uint64_t *buf = (uint64_t *)*buffer;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = (uint64_t)(low + (int64_t)(xorshift128plus() % range));
            }
            break;
        }
        case DTYPE_U32: {
            uint32_t *buf = (uint32_t *)*buffer;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = (uint32_t)(low + (int64_t)(xorshift128plus() % range));
            }
            break;
        }
        case DTYPE_U16: {
            uint16_t *buf = (uint16_t *)*buffer;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = (uint16_t)(low + (int64_t)(xorshift128plus() % range));
            }
            break;
        }
        case DTYPE_U8: {
            uint8_t *buf = (uint8_t *)*buffer;
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = (uint8_t)(low + (int64_t)(xorshift128plus() % range));
            }
            break;
        }
        default:
            free(*buffer);
            *buffer = NULL;
            REPORT_ERROR(ERR_DTYPE_MISMATCH, "Unsupported dtype in helper_buffer_randint");
            return false;
    }

    return true;
}


bool list_FunctionPtr_create(List_FunctionPtr **out, int initial_capacity) {
    if (!out) {
         return false; 
    }
    
    List_FunctionPtr *list = malloc(sizeof(List_FunctionPtr));
    if (!list) { 
        return false; }

    list->data = malloc(initial_capacity * sizeof(FunctionPtr));
    if (!list->data) {
      // WRITE("DEBUG: Failed to malloc list data\n")
        free(list);
        return false;
    }

    list->size = 0;
    list->capacity = initial_capacity;
    *out = list;
  // WRITE("DEBUG: End of list_functionPtr_create, returning true\n")
    return true;
}

bool list_FunctionPtr_push(List_FunctionPtr *list, FunctionPtr value) {
  // WRITE("DEBUG: Start of list_functionPtr_push\n")
    if (!list) { return false; }

    if (list->size >= list->capacity) {
        int new_cap = list->capacity * 2;
        FunctionPtr *new_data = realloc(list->data, new_cap * sizeof(FunctionPtr));
        if (!new_data) {return false; }
        list->data = new_data;
        list->capacity = new_cap;
    }

    list->data[list->size] = value;
    list->size++;
  // WRITE("DEBUG: End of list_functionPtr_push, returning true\n")
    return true;
}

bool list_FunctionPtr_get(List_FunctionPtr *list, int index, FunctionPtr *out) {
  // WRITE("DEBUG: Start of list_functionPtr_get\n")
    if (!list || index < 0 || index >= list->size || !out) {return false;} 
    *out = list->data[index];
  // WRITE("DEBUG: End of list_functionPtr_get, returning true\n")
    return true;
}

bool list_FunctionPtr_free(List_FunctionPtr *list) {
  // WRITE("DEBUG: Start of list_functionPtr_free\n")
    if (!list) { return false; }
    free(list->data);
    free(list);
  // WRITE("DEBUG: End of list_functionPtr_free\n")
    return true;
}
