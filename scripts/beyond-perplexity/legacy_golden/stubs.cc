#include "ggml.h"
#include <cstdio>
#include <cstdlib>
void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fprintf(stderr, "%s:%d: abort\n", file, line);
    abort();
}
size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    return (size_t) ne * ggml_type_size(type) / ggml_blck_size(type);
}
size_t ggml_type_size(enum ggml_type type) {
    static const size_t sz[] = {0,2,4,4,2,4,4,2,2,4,4,2,4,4,2,2,4,4,4,4,4,4,4,4,4,4,4,4,2,2,18,34,18,34,18,34,18,34,36,32,8,20,18,14,26};
    return sz[type];
}
int64_t ggml_blck_size(enum ggml_type type) {
    static const int bs[] = {0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,256,256,32,32,32,32,32,32,32,32,256,256,32,64,64,32,64};
    return bs[type];
}
const char * ggml_type_name(enum ggml_type type) {
    switch (type) { case GGML_TYPE_Q3_0: return "q3_0"; case GGML_TYPE_Q3_0_64: return "q3_0_64"; case GGML_TYPE_Q2_0: return "q2_0"; default: return "?"; }
}
