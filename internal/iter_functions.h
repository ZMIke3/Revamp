#include "Iter.h"
#include "arraylist.h"
#include <stdlib.h>


bool element_wise_iter_create(Iter **iter, ArrayList *list, IterCtx *ctx);

bool element_wise_iter_next(Iter *iter);

bool element_wise_iter_run(Iter *iter, kernel_fn func);

bool element_wise_iter_next_ele(Iter *iter);



bool reduction_iter_create(Iter **iter, ArrayList *list, IterCtx *ctx);

bool reduction_iter_next(Iter *iter);

bool reduction_iter_run(Iter *iter, kernel_fn func);



bool matmul_iter_create(Iter **iter, ArrayList *list, IterCtx *ctx);

bool matmul_iter_next(Iter *iter);

bool matmul_iter_run(Iter *iter, kernel_fn func);

