#include "dtypes.h"



bool dtypes_create_dtype(dtypes **out_type, dtype type) {
    *out_type = malloc(sizeof(dtypes));
    if(*out_type == NULL) {
        REPORT_ERROR(ERR_MALLOC_FAILED, "Failed to allocate memory of size %zu for dtype", sizeof(dtypes));
        return false;
    }

    size_t out_dtype_size;
    if (dtypes_sizeof_type(&out_dtype_size, type) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: dtypes_sizeof_type");
        free(*out_type);
        return false;
    }

    (*out_type)->type = type;
    (*out_type)->elesize_in_bytes = out_dtype_size;

    return true;

}

bool dtypes_sizeof_type(size_t *out_dtype_size, dtype type) {
    switch (type) {
        case DTYPE_I8:   *out_dtype_size = sizeof(int8_t);   return true;
        case DTYPE_I16:  *out_dtype_size = sizeof(int16_t);  return true;
        case DTYPE_I32:  *out_dtype_size = sizeof(int32_t);  return true;
        case DTYPE_I64:  *out_dtype_size = sizeof(int64_t);  return true;

        case DTYPE_U8:   *out_dtype_size = sizeof(uint8_t);  return true;
        case DTYPE_U16:  *out_dtype_size = sizeof(uint16_t); return true;
        case DTYPE_U32:  *out_dtype_size = sizeof(uint32_t); return true;
        case DTYPE_U64:  *out_dtype_size = sizeof(uint64_t); return true;

        case DTYPE_F32:  *out_dtype_size = sizeof(float);    return true;
        case DTYPE_F64:  *out_dtype_size = sizeof(double);   return true;

        case DTYPE_BOOL: *out_dtype_size = sizeof(bool);     return true;

        case DTYPE_COUNT:
        default:
            REPORT_ERROR(ERR_INPUT_OUT_OF_BOUNDS, "input must be: type <= %d", DTYPE_COUNT);
            return false; 
    }
}

bool dtypes_promote_dtype(dtype *out_type, dtype type_1, dtype type_2, bool is_signed_1, bool is_signed_2) {

    if (type_1 >= DTYPE_COUNT || type_2 >= DTYPE_COUNT) {
        REPORT_ERROR(ERR_INPUT_OUT_OF_BOUNDS, "input must be: type_1 <= %d && type_2 <= %d", DTYPE_COUNT, DTYPE_COUNT);
        return false;
    }

    if (type_1 == type_2) { *out_type = type_1;  return true; }
    

    if (type_1 == DTYPE_F64 || type_2 == DTYPE_F64) {
        *out_type = DTYPE_F64;
        return true;
    }

    if (type_1 == DTYPE_F32 || type_2 == DTYPE_F32) {
        *out_type = DTYPE_F32;
        return true;
    }

    size_t size_1;
    size_t size_2;

    dtypes_sizeof_type(&size_1, type_1);
    dtypes_sizeof_type(&size_2, type_2);

    if (size_1 == size_2) {
        if (is_signed_1 && !is_signed_2) {
            switch (size_1) {
                case 1: *out_type = DTYPE_I16; return true;
                case 2: *out_type = DTYPE_I32; return true;
                case 4: *out_type = DTYPE_I64; return true;
                case 8: *out_type =  DTYPE_F64; return true;
            }
        } else if (!is_signed_1 && is_signed_2) {
            switch (size_2) {
                case 1: *out_type = DTYPE_I16; return true;
                case 2: *out_type = DTYPE_I32; return true;
                case 4: *out_type = DTYPE_I64; return true;
                case 8: *out_type =  DTYPE_F64; return true;
            }
        }
    }
    
    if (size_1 > size_2) {
        *out_type = type_1;
        return true;
    } else if (size_2 > size_1) {
        *out_type = type_2;
        return true;
    }
    
    *out_type = is_signed_1 ? type_1 : type_2;

    return true;
    
    
}

bool dtypes_dtype_is_signed(dtype type, int *is_signed) {

  switch (type) {
        case DTYPE_I8:   *is_signed = 1;  return true;
        case DTYPE_I16:  *is_signed = 1;  return true;
        case DTYPE_I32:  *is_signed = 1;  return true;
        case DTYPE_I64:  *is_signed = 1;  return true;

        case DTYPE_U8:   *is_signed = 0;   return true;
        case DTYPE_U16:  *is_signed = 0;   return true;
        case DTYPE_U32:  *is_signed = 0;   return true;
        case DTYPE_U64:  *is_signed = 0;   return true;

        case DTYPE_F32:  *is_signed = 1;   return true;
        case DTYPE_F64:  *is_signed = 1;   return true;

        case DTYPE_BOOL: *is_signed = 0;   return true;

        case DTYPE_COUNT:
        default:
            REPORT_ERROR(ERR_INPUT_OUT_OF_BOUNDS, "input must be: type <= %d", DTYPE_COUNT);
            return false; 
    }
}

bool dtypes_fast_contiguous_cast(void* src_ptr, dtype src_dtype, void* dst_ptr, dtype dst_dtype, uint64_t count) {
    switch (src_dtype) {
        case DTYPE_I8:   DTYPES_DISPATCH_DST_TYPE(int8_t,   src_ptr, dst_ptr, dst_dtype, count); return true;
        case DTYPE_I16:  DTYPES_DISPATCH_DST_TYPE(int16_t,  src_ptr, dst_ptr, dst_dtype, count); return true;
        case DTYPE_I32:  DTYPES_DISPATCH_DST_TYPE(int32_t,  src_ptr, dst_ptr, dst_dtype, count); return true;
        case DTYPE_I64:  DTYPES_DISPATCH_DST_TYPE(int64_t,  src_ptr, dst_ptr, dst_dtype, count); return true;
        case DTYPE_U8:   DTYPES_DISPATCH_DST_TYPE(uint8_t,  src_ptr, dst_ptr, dst_dtype, count); return true;
        case DTYPE_U16:  DTYPES_DISPATCH_DST_TYPE(uint16_t, src_ptr, dst_ptr, dst_dtype, count); return true;
        case DTYPE_U32:  DTYPES_DISPATCH_DST_TYPE(uint32_t, src_ptr, dst_ptr, dst_dtype, count); return true;
        case DTYPE_U64:  DTYPES_DISPATCH_DST_TYPE(uint64_t, src_ptr, dst_ptr, dst_dtype, count); return true;
        case DTYPE_F32:  DTYPES_DISPATCH_DST_TYPE(float,    src_ptr, dst_ptr, dst_dtype, count); return true;
        case DTYPE_F64:  DTYPES_DISPATCH_DST_TYPE(double,   src_ptr, dst_ptr, dst_dtype, count); return true;
        case DTYPE_BOOL: DTYPES_DISPATCH_DST_TYPE(bool,     src_ptr, dst_ptr, dst_dtype, count); return true;
        case DTYPE_COUNT:
        default:
            REPORT_ERROR(ERR_INPUT_OUT_OF_BOUNDS, "input must be: src_dtype <= %d", DTYPE_COUNT);
            return false;
    }
}

bool dtypes_free_dtype(dtypes *type_obj) {
    free(type_obj);
}


