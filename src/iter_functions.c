#include "iter_functions.h"


bool element_wise_iter_create(Iter **iter, ArrayList *list, IterCtx *ctx) {
    ElementwiseIter *e = malloc(sizeof(ElementwiseIter));

    if (!e || !list) {
        return false;
    }

    e->base.type = ITER_ELEMENTWISE;
    e->base.next = element_wise_iter_next;
    e->base.run = element_wise_iter_run;
    e->base.next_ele = element_wise_iter_next_ele;

    uint64_t *e_shape;
    uint64_t e_ndim;

    if (!list->list_broadcasted) {
       if (arraylist_broadcast_and_max_ndim(list, &e_shape, &e_ndim) == false) { return false; }
    }


    e->nops = list->noe;
    e->ndim = list->broadcasted_ndim;
    e->shape = malloc(sizeof(uint64_t) * e->ndim);
    memcpy(e->shape, list->broadcasted_shape, sizeof(uint64_t) * e->ndim);

    if (arraylist_broadcast_stride_from_shape(list,  e->shape, e->ndim, &e->strides) == false) { 
        return false; 
    }

    e->baseptrs = malloc(sizeof(char*) * e->nops);
    e->dataptrs = malloc(sizeof(char*) * e->nops);

    if (!e->baseptrs || !e->dataptrs) {
        return false; 
    }

    for (int i = 0; i < e->nops; i++) {
        e->baseptrs[i] = list->arrays[i]->data;
        e->dataptrs[i] = list->arrays[i]->data;
    }

    e->backstrides = malloc(sizeof(int64_t *) * e->nops);
    if (!e->backstrides) {
        return false; 
    }

    e->coordinates = calloc(sizeof(uint64_t), e->ndim);
    if (!e->coordinates) { 
        return false; 
    }

    for (int ar = 0; ar < e->nops; ar++) {    
        e->backstrides[ar] = malloc(sizeof(int64_t) * e->ndim);
        if (!e->backstrides[ar]) { return false; }

        for (uint64_t dim = 0; dim < e->ndim; dim++) {
             int64_t backstride = e->strides[ar][dim] * ((int64_t)e->shape[dim] - 1);
             e->backstrides[ar][dim] = backstride;
        }

    }

    for (int op = 0; op < e->nops; op++) {
        int64_t offset = e->ndim - list->arrays[op]->ndim;
        for (int i = 0; i < e->ndim; i++) {
            int64_t arr_dim = i - offset;
            if (arr_dim >= 0) {
                if (list->arrays[op]->shape[arr_dim] == 1 && e->shape[i] > 1) {
                    e->strides[op][i] = 0;
                    e->backstrides[op][i] = 0;
                }
            } else {
                e->strides[op][i] = 0;
                e->backstrides[op][i] = 0;
            }
        }
    }

    e->counter = 0;

    e->inner_strides = malloc(sizeof(int64_t) * e->nops);

    for (int i = 0; i < e->nops; i++) {
        e->inner_strides[i] = e->strides[i][e->ndim - 1];
    }

    e->total_size = 1;
    for (uint64_t dim = 0; dim < e->ndim; dim++) {
        e->total_size *= e->shape[dim]; 
    }

    if (e->ndim > 1) {
        e->inner_loop_size = e->shape[e->ndim - 1];
        e->inner_ndim = 1;
        e->outer_ndim = e->ndim - 1;
    } else {
        e->inner_ndim = 0;
        e->outer_ndim = 0;
        e->inner_loop_size = e->total_size;
    }
    
   *iter = (Iter *) e;

   return true;


}

bool element_wise_iter_next(Iter *iter) {
    if (!iter) { 
        return false; 
    }

    ElementwiseIter *e = (ElementwiseIter *) iter;

    e->counter += e->inner_loop_size;

    if (e->counter >= e->total_size) {
        return false;
    }

    for (int64_t dim = e->outer_ndim - 1; dim >= 0; dim--) {
        if (e->coordinates[dim] + 1 < e->shape[dim]) {
            
            e->coordinates[dim]++;
 
            for (int arr = 0; arr < e->nops; arr++) {
                e->dataptrs[arr] += e->strides[arr][dim];
            }
            return true;
        }

        e->coordinates[dim] = 0;
        for (int arr = 0; arr < e->nops; arr++) {
            e->dataptrs[arr] -= e->backstrides[arr][dim];

        }
        
    }
    
    return false;

}

bool element_wise_iter_run(Iter *iter, kernel_fn func) {
    if (!iter) { 
        return false; 
    }

    ElementwiseIter *e = (ElementwiseIter *) iter;

    kernel_info info = {
        .dataptrs = e->dataptrs,
        .inner_loop_size = e->inner_loop_size,
        .inner_strides = e->inner_strides,
    };

    if (e->total_size == 0) {
        return true; 
    }

    do {
        
        func(&info);

    } while (iter->next(iter));

    return true;

}

bool element_wise_iter_next_ele(Iter *iter) {
    if (!iter) return false;

    ElementwiseIter *e = (ElementwiseIter *) iter;

    if (e->counter + 1 >= e->total_size) {
        return false;
    }
    
    e->counter++;

    for (int64_t dim = e->ndim - 1; dim >= 0; dim--) {

        if (e->coordinates[dim] + 1 < e->shape[dim]) {
            e->coordinates[dim]++;
            
            for (int arr = 0; arr < e->nops; arr++) {
                e->dataptrs[arr] += e->strides[arr][dim];
            }

            return true;

        }
        
        e->coordinates[dim] = 0;
        for (int arr = 0; arr < e->nops; arr++) {
            e->dataptrs[arr] -= e->backstrides[arr][dim];
        }
        
    }

    return false;

}


bool reduction_iter_create(Iter **iter, ArrayList *list, IterCtx *ctx) {
    ReductionIter *r = malloc(sizeof(ReductionIter));

    if (!r || !list) {
        return false;
    }
    // printf("1");
    r->base.type = ITER_REDUCTION;
    r->base.next = reduction_iter_next;
    r->base.run = reduction_iter_run;


    r->ndim = ctx->input_ndim;
    r->nops = list->noe;

    r->baseptrs = malloc(sizeof(char*) * r->nops);
    r->dataptrs = malloc(sizeof(char*) * r->nops);

    if (!r->baseptrs || !r->dataptrs) {
        return false; 
    }
    // printf("2");


    for (int i = 0; i < r->nops; i++) {
        r->baseptrs[i] = list->arrays[i]->data;
        r->dataptrs[i] = list->arrays[i]->data;
    }

    r->shape = malloc(sizeof(uint64_t) * r->ndim);
    r->coordinates = calloc(sizeof(uint64_t), r->ndim);
    r->backstrides = malloc(sizeof(int64_t *) * r->nops);
    // printf("3");


    for (uint64_t i = 0; i < r->ndim; i++) {
        r->shape[i] = ctx->input_shape[ctx->reduction_perm[i]];
    }

    if (arraylist_broadcast_stride_from_shape(list, r->shape, r->ndim, &r->strides) == false) { 
        return false; 
    }
    // printf("4");

    for (int ar = 0; ar < r->nops; ar++) {
        int64_t *old = r->strides[ar];
        int64_t *new = malloc(sizeof(int64_t) * r->ndim);

        for (uint64_t i = 0; i < r->ndim; i++) {
            new[i] = old[ctx->reduction_perm[i]];
        }
        r->strides[ar] = new;
    }
    // printf("5");



    for (int ar = 0; ar < r->nops; ar++) {
        r->backstrides[ar] = malloc(sizeof(int64_t) * r->ndim);
        if (!r->backstrides[ar]) { return false; }

        for (uint64_t d = 0; d < r->ndim; d++) {
            r->backstrides[ar][d] = r->strides[ar][d] * ((int64_t)r->shape[d] - 1);
        }
    }
    // printf("6");

    r->reduce_shape = malloc(sizeof(int64_t) * ctx->num_reduced);
    r->reduce_strides = malloc(sizeof(int64_t *) * r->nops);
        
    for (int ar = 0; ar < r->nops; ar++) {
        r->reduce_strides[ar] = malloc(sizeof(int64_t) * ctx->num_reduced);
    }


    for (int op = 0; op < ctx->num_reduced; op++) {
        r->reduce_shape[op] = r->shape[r->ndim - ctx->num_reduced + op];
    }
    // printf("7");


    for (int ar = 0; ar < r->nops; ar++) {
        for (int l = 0; l < ctx->num_reduced; l++) {
            r->reduce_strides[ar][l] = r->strides[ar][r->ndim - ctx->num_reduced + l];
        }
    }

    int out_idx = r->nops - 1;
    for (uint64_t i = r->ndim - ctx->num_reduced; i < r->ndim; i++) {
        r->strides[out_idx][i] = 0;
    }
    // printf("8");


    int64_t curr_stride = 1;
    for (int i = ctx->num_reduced - 1; i > 0; i--) {
        r->flat_strides[i] = curr_stride;
        curr_stride *= i * r->reduce_shape[i];
    }

    r->total_size = 1;
    for(uint64_t i = 0; i < r->ndim; i++) {
        r->total_size *= r->shape[i];
    }    
    // printf("9");


    r->inner_loop_size = 1;
    for (uint64_t i = r->ndim - ctx->num_reduced; i < r->ndim; i++) {
        r->inner_loop_size *= r->shape[i];
    }

    r->ndim = r->ndim - ctx->num_reduced;
    r->outer_ndim = r->ndim;
    r->inner_ndim = ctx->num_reduced;
    r->counter = 0;
    // printf("10");


    *iter = (Iter *) r;

   return true;

}

bool reduction_iter_next(Iter *iter) {
    if (!iter) { 
        return false; 
    }

    ReductionIter *r = (ReductionIter *) iter;

    r->counter += r->inner_loop_size;

    if (r->counter >= r->total_size) {
        return false;
    }

    for (int64_t dim = r->outer_ndim - 1; dim >= 0; dim--) {
        if (r->coordinates[dim] + 1 < r->shape[dim]) {
            
            r->coordinates[dim]++;
 
            for (int arr = 0; arr < r->nops; arr++) {
                r->dataptrs[arr] += r->strides[arr][dim];
            }
            return true;
        }

        r->coordinates[dim] = 0;
        for (int arr = 0; arr < r->nops; arr++) {
            r->dataptrs[arr] -= r->backstrides[arr][dim];

        }
        
    }
    
    return false;

}

bool reduction_iter_run(Iter *iter, kernel_fn func) {
    if (!iter) { 
        return false; 
    }

    ReductionIter *r = (ReductionIter *) iter;

    kernel_info info = {
        .dataptrs = r->dataptrs,
        .inner_loop_size = r->inner_loop_size,
        .rndims = r->ndim,
        .reduce_shape = r->reduce_shape,
        .reduce_strides = r->reduce_strides,
        .flat_strides = r->flat_strides
    };

    if (r->total_size == 0) {
        return true; 
    }

    do {
        
        func(&info);

    } while (iter->next(iter));

    return true;

}


bool matmul_iter_create(Iter **iter, ArrayList *list, IterCtx *ctx) {
    if (!iter || !list || !ctx) return false;
    
    MatmulIter *m = malloc(sizeof(MatmulIter));
    if (!m) return false;

    m->base.type = ITER_BATCHED_MATMUL;
    m->base.next = matmul_iter_next;
    m->base.run  = matmul_iter_run;

    m->nops = list->noe;

    m->arr_a_row_mm = ctx->arr_a_row_mm;
    m->arr_a_col_mm = ctx->arr_a_col_mm;
    m->arr_b_col_mm = ctx->arr_b_col_mm;

    // shape comes from output array (index 2)
    m->ndim  = list->arrays[2]->ndim;
    m->shape = malloc(sizeof(uint64_t) * m->ndim);
    if (!m->shape) goto fail;
    memcpy(m->shape, list->arrays[2]->shape, sizeof(uint64_t) * m->ndim);

    m->outer_ndim = (m->ndim >= 2) ? m->ndim - 2 : 0;
    m->inner_ndim = 2;
    // printf("M_ndim_1: %llu\n", m->ndim);
    

    // batch strides
    m->strides     = calloc(m->nops, sizeof(int64_t*));
    m->backstrides = calloc(m->nops, sizeof(int64_t*));
    if (!m->strides || !m->backstrides) goto fail;

    for (int ar = 0; ar < m->nops; ar++) {
        m->strides[ar]     = calloc(m->outer_ndim, sizeof(int64_t));
        m->backstrides[ar] = calloc(m->outer_ndim, sizeof(int64_t));
        if (!m->strides[ar] || !m->backstrides[ar]) goto fail;

        array *arr = list->arrays[ar];
        if (!arr) goto fail;

        int64_t arr_batch_ndim = (arr->ndim >= 2) ? arr->ndim - 2 : 0;
        int64_t offset = (int64_t)m->outer_ndim - arr_batch_ndim;

        for (int64_t dim = 0; dim < (int64_t)m->outer_ndim; dim++) {
            int64_t arr_dim = dim - offset; // What does it mean if this is negative?
            uint64_t shape_here = (arr_dim >= 0) ? arr->shape[arr_dim] : 1;

            if (arr_dim >= 0 && shape_here != 1) { // Why no shape of 1 in matmul?
                m->strides[ar][dim]     = arr->stride[arr_dim];
                m->backstrides[ar][dim] = arr->stride[arr_dim] * ((int64_t)m->shape[dim] - 1);

            }
            // broadcast/out-of-bounds dims stay 0 from calloc
            // printf("ar:%d at dim %lld, strides:%lld\n", ar, dim, m->strides[ar][dim]);
            // printf("M_outer_ndim:%llu\n", m->outer_ndim);
            // printf("M_ndim: %llu\n", m->ndim);

        }
    }

    // baseptrs / dataptrs
    m->baseptrs = malloc(sizeof(char*) * m->nops);
    m->dataptrs = malloc(sizeof(char*) * m->nops);
    if (!m->baseptrs || !m->dataptrs) goto fail;

    for (int i = 0; i < m->nops; i++) {
        if (!list->arrays[i] || !list->arrays[i]->data) goto fail;
        m->baseptrs[i] = list->arrays[i]->data;
        m->dataptrs[i] = list->arrays[i]->data;
    }

    m->coordinates = calloc(m->outer_ndim, sizeof(uint64_t));
    if (!m->coordinates) goto fail;

    m->total_size = 1;
for (uint64_t d = 0; d < m->outer_ndim; d++)
    m->total_size *= m->shape[d];

   // printf("Total Size: %llu\n", m->total_size);
    
    m->inner_loop_size = 1;

    for (uint64_t d = m->outer_ndim; d < m->ndim; d++) {
        m->inner_loop_size *= m->shape[d];
    }

  //  printf("Inner loop Size: %llu\n", m->inner_loop_size);


    m->counter = 0;
   // m->total_size = m->total_size / m->inner_loop_size;
 //   printf("Total Size: %llu\n", m->total_size);

    *iter = (Iter *) m;
    return true;

fail:
    for (int ar = 0; ar < m->nops; ar++) {
        if (m->strides)     free(m->strides[ar]);
        if (m->backstrides) free(m->backstrides[ar]);
    }
    free(m->strides);
    free(m->backstrides);
    free(m->shape);
    free(m->baseptrs);
    free(m->dataptrs);
    free(m->coordinates);
    free(m);
    *iter = NULL;
    return false;
}

bool matmul_iter_next(Iter *iter) {
    if (!iter) { 
        return false; 
    }

    MatmulIter *m = (MatmulIter *) iter;

    m->counter += 1;

    if (m->counter >= m->total_size) {
        return false;
    }

    for (int64_t dim = m->outer_ndim - 1; dim >= 0; dim--) {
        if (m->coordinates[dim] + 1 < m->shape[dim]) {
            
            m->coordinates[dim]++;
 
            for (int arr = 0; arr < m->nops; arr++) {
                m->dataptrs[arr] += m->strides[arr][dim];
            }
            return true;
        }

        m->coordinates[dim] = 0;
        for (int arr = 0; arr < m->nops; arr++) {
            m->dataptrs[arr] -= m->backstrides[arr][dim];

        }
        
    }
    
    return false;

}

bool matmul_iter_run(Iter *iter, kernel_fn func) {
    if (!iter) { 
        return false; 
    }

    MatmulIter *m = (MatmulIter *) iter;

    kernel_info info = {
        .dataptrs = m->dataptrs,
        .inner_loop_size = m->inner_loop_size,
        .arr_a_col_mm = m->arr_a_col_mm,
        .arr_a_row_mm = m->arr_a_row_mm,
        .arr_b_col_mm = m->arr_b_col_mm,
    };

    if (m->total_size == 0) {
        return true; 
    }

    do {
        
        func(&info);

    } while (iter->next(iter));

    return true;

}
