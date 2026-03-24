#include "Functions.h"

bool array_add_backward(Function *fn) {
  if (!fn || !fn->output || !fn->inputs[0] || !fn->inputs[1]) {
      REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Function or it's inputs or output fields are not valid");
      return false;
    }

    array *in0_grad = fn->inputs[0]->grad;
    array *in1_grad = fn->inputs[1]->grad;
    array *out_grad = fn->output->grad;

    OpInstr instr = {
        .op = OP_ADD_BACKWARDS, 
        .inArity = 3, 
        .outArity = 0,        
        .backward_pass = true
    };

    ArrayList *list;

   if (arraylist_create(&list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_create");
        return false;
    }

    if (!arraylist_append(list, in0_grad)) {  
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    if (!arraylist_append(list, in1_grad)) {  
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    if (!arraylist_append(list, out_grad)) {  
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    OperationContext *ctx;

    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }


   if (ctx->iter->run(ctx->iter, ctx->fn) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: Iter_run_kernel");
        return false;
   }


  if (operation_context_free(ctx, false) == false) {
      REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
   }
    
    return true;
}

bool array_subtract_backward(Function *fn) {
    if (!fn || !fn->output || !fn->inputs[0] || !fn->inputs[1]) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Function or it's inputs or output fields are not valid");
        return false;
    }

    array *in0_grad = fn->inputs[0]->grad;
    array *in1_grad = fn->inputs[1]->grad;
    array *out_grad = fn->output->grad;

    OpInstr instr = {
        .op = OP_SUBTRACT_BACKWARDS,
        .inArity = 3, 
        .outArity = 0,        
        .backward_pass = true
    };

    ArrayList *list;

    if (arraylist_create(&list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_create");
        return false;
    }

    if (!arraylist_append(list, in0_grad)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    if (!arraylist_append(list, in1_grad)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    if (!arraylist_append(list, out_grad)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    OperationContext *ctx;

    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   if (ctx->iter->run(ctx->iter, ctx->fn) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: Iter_run_kernel");
        return false;
   }

    if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    return true;
}

bool array_multiply_backward(Function *fn) {
    if (!fn || !fn->output || !fn->inputs[0] || !fn->inputs[1]) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Function or it's inputs or output fields are not valid");
        return false;
    }

    array *in0_grad = fn->inputs[0]->grad;
    array *in1_grad = fn->inputs[1]->grad;
    array *out_grad = fn->output->grad;
    array *in0_val = fn->inputs[0]->data;
    array *in1_val = fn->inputs[1]->data;

    OpInstr instr = {
        .op = OP_MULTIPLY_BACKWARDS,
        .inArity = 3, 
        .outArity = 0,        
        .backward_pass = true
    };

    ArrayList *list;

    if (arraylist_create(&list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_create");
        return false;
    }

    if (!arraylist_append(list, in0_grad)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    if (!arraylist_append(list, in1_grad)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    if (!arraylist_append(list, in0_val)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    if (!arraylist_append(list, in1_val)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    if (!arraylist_append(list, out_grad)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    OperationContext *ctx;

    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   if (ctx->iter->run(ctx->iter, ctx->fn) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: Iter_run_kernel");
        return false;
   }

    if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    return true;
}

bool array_divide_backward(Function *fn) {
    if (!fn || !fn->output || !fn->inputs[0] || !fn->inputs[1]) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Function or it's inputs or output fields are not valid");
        return false;
    }

    array *in0_grad = fn->inputs[0]->grad;
    array *in1_grad = fn->inputs[1]->grad;
    array *out_grad = fn->output->grad;
    array *in0_val = fn->inputs[0]->data;
    array *in1_val = fn->inputs[1]->data;

    OpInstr instr = {
        .op = OP_DIVIDE_BACKWARDS,
        .inArity = 5, 
        .outArity = 0,        
        .backward_pass = true
    };

    ArrayList *list;

    if (arraylist_create(&list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_create");
        return false;
    }

    if (!arraylist_append(list, in0_grad)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    if (!arraylist_append(list, in1_grad)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    if (!arraylist_append(list, in0_val)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    if (!arraylist_append(list, in1_val)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    if (!arraylist_append(list, out_grad)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    OperationContext *ctx;

    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   if (ctx->iter->run(ctx->iter, ctx->fn) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: Iter_run_kernel");
        return false;
   }

    if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    return true;
}

bool array_power_backward(Function *fn) {
    if (!fn || !fn->output || !fn->inputs[0] || !fn->inputs[1]) {
        REPORT_ERROR(ERR_INPUT_IS_NOT_INIT, "Function or it's inputs or output fields are not valid");
        return false;
    }

    array *in0_grad = fn->inputs[0]->grad;
    array *in1_grad = fn->inputs[1]->grad;
    array *out_grad = fn->output->grad;
    array *out = fn->output->data;
    array *in0_val = fn->inputs[0]->data;
    array *in1_val = fn->inputs[1]->data;

    OpInstr instr = {
        .op = OP_POWER_BACKWARDS,
        .inArity = 5, 
        .outArity = 0,        
        .backward_pass = true
    };

    ArrayList *list;

    if (arraylist_create(&list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_create");
        return false;
    }

    if (!arraylist_append(list, in0_grad)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    if (!arraylist_append(list, in1_grad)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    if (!arraylist_append(list, in0_val)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    if (!arraylist_append(list, in1_val)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    if (!arraylist_append(list, out)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    if (!arraylist_append(list, out_grad)) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: arraylist_append");
        return false;
    }

    OperationContext *ctx;

    if (configure_operation(&ctx, &instr, list) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: configure_operation");
        return false;
    }

   if (ctx->iter->run(ctx->iter, ctx->fn) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: Iter_run_kernel");
        return false;
   }

    if (operation_context_free(ctx, false) == false) {
        REPORT_ERROR(ERR_FUNC_CALL_FAILED, "func: operation_context_free");
        return false;
    }

    return true;
}
