#ifndef DTYPES_H
#define DTYPES_H

#include <stdbool.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdio.h>
#include "Error.h"

typedef enum {
    DTYPE_I8,
    DTYPE_I16, 
    DTYPE_I32,
    DTYPE_I64,
    DTYPE_U8,
    DTYPE_U16,
    DTYPE_U32,
    DTYPE_U64,
    DTYPE_F32,
    DTYPE_F64,
    DTYPE_BOOL,
    DTYPE_NONE,
    DTYPE_COUNT
} dtype;

typedef struct dtypes {
    bool is_signed;
    dtype type;
    size_t elesize_in_bytes;
} dtypes;


#define DTYPES_CAST_INNER_LOOP(src_type, dst_type, count, src_ptr, dst_ptr) \
    for (uint64_t i = 0; i < count; i++) {                           \
        ((dst_type*)dst_ptr)[i] = (dst_type)(((src_type*)src_ptr)[i]); \
    }

#define DTYPES_DISPATCH_DST_TYPE(src_type, src_ptr, dst_ptr, dst_dtype, count) \
    switch (dst_dtype) {                                                       \
        case DTYPE_I8:  DTYPES_CAST_INNER_LOOP(src_type, int8_t,   count, src_ptr, dst_ptr); break; \
        case DTYPE_I16: DTYPES_CAST_INNER_LOOP(src_type, int16_t,  count, src_ptr, dst_ptr); break; \
        case DTYPE_I32: DTYPES_CAST_INNER_LOOP(src_type, int32_t,  count, src_ptr, dst_ptr); break; \
        case DTYPE_I64: DTYPES_CAST_INNER_LOOP(src_type, int64_t,  count, src_ptr, dst_ptr); break; \
        case DTYPE_U8:  DTYPES_CAST_INNER_LOOP(src_type, uint8_t,  count, src_ptr, dst_ptr); break; \
        case DTYPE_U16: DTYPES_CAST_INNER_LOOP(src_type, uint16_t, count, src_ptr, dst_ptr); break; \
        case DTYPE_U32: DTYPES_CAST_INNER_LOOP(src_type, uint32_t, count, src_ptr, dst_ptr); break; \
        case DTYPE_U64: DTYPES_CAST_INNER_LOOP(src_type, uint64_t, count, src_ptr, dst_ptr); break; \
        case DTYPE_F32: DTYPES_CAST_INNER_LOOP(src_type, float,    count, src_ptr, dst_ptr); break; \
        case DTYPE_F64: DTYPES_CAST_INNER_LOOP(src_type, double,   count, src_ptr, dst_ptr); break; \
        case DTYPE_BOOL:DTYPES_CAST_INNER_LOOP(src_type, bool,     count, src_ptr, dst_ptr); break; \
        default: break;                                                                             \
    }


bool dtypes_create_dtype(dtypes **out_type, dtype type);

bool dtypes_sizeof_type(size_t *out_dtype_size, dtype type);

bool dtypes_promote_dtype(dtype *out_type, dtype type_1, dtype type_2, bool is_signed_1, bool is_signed_2);

bool dtypes_dtype_is_signed(dtype type, int *is_signed);

bool dtypes_fast_contiguous_cast(void* src_ptr, dtype src_dtype, void* dst_ptr, dtype dst_dtype, uint64_t count);

bool dtypes_free_dtype(dtypes *type_obj);


#endif // DTYPES_H
