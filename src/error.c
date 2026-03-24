#include "Error.h"


error_ctx err_t = {.top = -1};


const char* error_get_msg(error_id code) {
    switch(code) {
#define X(id, msg) case id: return msg;
        ERROR_TABLE(X)
#undef X
        default: return "Unknown Error";
    }
}


void error_stack_print() {
    printf("--- REVAMP ERROR STACK ---\n");
    //fprintf(stderr, "--- REVAMP ERROR STACK ---\n");
    while (err_t.top >= 0) {
        error *e = &err_t.stack[err_t.top];
        
        fprintf(stderr, "[%s] %s:%d\n", error_get_msg(e->code), e->file, e->line);
        fprintf(stderr, "  Detail: %s\n", e->custom_msg);
        
        err_t.top--;
    }
}
