// Golden vector generator for beyond-perplexity reference formats.
//
// Compiles the LEGACY implementation's ggml-quants.c verbatim (read-only
// include from the frozen legacy repo) and emits packed bytes and dequant
// values for the q2_0 (uniform), q3_0 and q3_0_64 formats.
//
// Usage:
//   LEGACY=/path/to/legacy/repo
//   gcc -O2 -I $LEGACY/ggml/include -I $LEGACY/ggml/src \
//       -c $LEGACY/ggml/src/ggml-quants.c -o ggml-quants.o
//   g++ -O2 -I $LEGACY/ggml/include -I $LEGACY/ggml/src \
//       golden_host.cc ggml-quants.o -o golden_host -lm
//   ./golden_host > golden_cpu.raw
//
// Output format (one JSON object per line):
//   {"type":"q3_0","case":"thresh","sub":2,"input":[...hex floats...],
//    "scale":"0x0000","packed":"...hex...","dequant":[...hex floats...]}

#include "ggml-quants.h"
#include "ggml-common.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>

static void emit_float(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    printf("{\"hex\":\"%08x\",\"f\":\"%.9g\"}", u, (double) f);
}

// ---- q3 family cases ----------------------------------------------------
// q3: d = max/-4 with max = -4 => d = 1, id = 1, x0 = x.
// code c covers x0 in [c-4.5, c-3.5). boundary between code c-1 and c at
// x0 = c-4.5. c = sub*2+1 -> transitions 0/1, 2/3, 4/5, 6/7

// q2 family (uniform q2_0): d = amax (positive). block anchored by
// v[qk-1] = amax, so x0 = x/amax in [-1, 1]. roundf(x0)+1 clamped 0..3:
// code 0 at x0 in [-1.5, -0.5), code 1 in [-0.5, 0.5), code 2 in [0.5, 1].
// boundaries at x = +-amax/2. code 3 unreachable (x0 <= 1).

static void gen_thresh_q3(float * v, int qk, int sub, float eps) {
    memset(v, 0, qk * sizeof(float));
    v[0] = -4.0f;
    v[1] = (float) (sub * 2 + 1) - 4.5f + eps;
}

static void gen_thresh_q2(float * v, int qk, float eps) {
    const float amax = 0.75f;
    memset(v, 0, qk * sizeof(float));
    v[qk - 1] = amax;
    v[1] = -0.5f * amax + eps;
    v[2] =  0.5f * amax + eps;
}

static void gen_all_codes_q3(float * v, int qk, int shift) {
    for (int j = 0; j < qk; ++j) {
        v[j] = (((j + shift) % 8) - 4) * -0.5f;
    }
}

static void gen_all_codes_q2(float * v, int qk, int shift) {
    const float amax = 0.75f;
    for (int j = 0; j < qk; ++j) {
        v[j] = (((j + shift) % 3) - 1.0f) * amax;
    }
    v[qk - 1] = amax; // anchor: forces amax
}

static void gen_all_positions_q3(float * v, int qk, int sub) {
    for (int j = 0; j < qk; ++j) v[j] = ((j * 7 + sub) % 8) - 4.0f;
}

static void gen_all_positions_q2(float * v, int qk, int sub) {
    const float amax = 0.75f;
    for (int j = 0; j < qk; ++j) v[j] = (((j * 7 + sub) % 3) - 1.0f) * amax;
    v[qk - 1] = amax;
}

// mode: 0 = q2 family (4 codes), 1 = q3 family (8 codes)
template <typename Block, int QK>
static void emit_type(const char * type_name, int mode,
                      void (*quant_ref)(const float *, Block *, int64_t),
                      void (*dequant_ref)(const Block *, float *, int64_t)) {
    float x[QK];
    float y[QK];
    Block b;

    const float eps[4] = {-1e-5f, 1e-5f, -1e-3f, 1e-3f};

    for (int ci = 0; ci < 8; ++ci) {
        for (int sub = 0; sub < 4; ++sub) {
            memset(x, 0, sizeof(x));
            switch (ci) {
                case 0: { // zeros (q2 also covers the degenerate d <= 0 path)
                } break;
                case 1: { // const_pos
                    for (int j = 0; j < QK; ++j) x[j] = 0.75f;
                } break;
                case 2: { // const_neg
                    for (int j = 0; j < QK; ++j) x[j] = -0.75f;
                } break;
                case 3: { // extrema
                    const float ext[4] = {0.001f, 3.0f, -2.5f, 1e-4f};
                    for (int j = 0; j < QK; ++j) x[j] = ext[(j + sub) % 4];
                } break;
                case 4: { // thresh: code boundaries, both sides
                    if (mode == 0) gen_thresh_q2(x, QK, eps[sub]);
                    else           gen_thresh_q3(x, QK, sub, eps[sub]);
                } break;
                case 5: { // all_codes at exact representable values
                    if (mode == 0) gen_all_codes_q2(x, QK, sub);
                    else           gen_all_codes_q3(x, QK, sub);
                } break;
                case 6: { // all_positions
                    if (mode == 0) gen_all_positions_q2(x, QK, sub);
                    else           gen_all_positions_q3(x, QK, sub);
                } break;
                case 7: { // random, fixed seed
                    srand(12345 + sub * 7919);
                    for (int j = 0; j < QK; ++j) {
                        x[j] = (float) (rand() % 20001) / 10000.0f - 1.0f;
                    }
                } break;
                default: break;
            }

            const char * case_name =
                ci == 0 ? "zeros" : ci == 1 ? "const_pos" : ci == 2 ? "const_neg" :
                ci == 3 ? "extrema" : ci == 4 ? "thresh" : ci == 5 ? "all_codes" :
                ci == 6 ? "all_positions" : "random";

            quant_ref(x, &b, QK);
            dequant_ref(&b, y, QK);

            printf("{\"type\":\"%s\",\"case\":\"%s\",\"sub\":%d,\"input\":[", type_name, case_name, sub);
            for (int j = 0; j < QK; ++j) { if (j) printf(","); emit_float(x[j]); }
            printf("],\"scale\":\"%04x\",\"packed\":\"", b.d);
            for (size_t i = 0; i < sizeof(b.qs); ++i) printf("%02x", b.qs[i]);
            printf("\",\"dequant\":[");
            for (int j = 0; j < QK; ++j) { if (j) printf(","); emit_float(y[j]); }
            printf("]}\n");
        }
    }
}

int main() {
    emit_type<block_q2_0, QK2_0>("q2_0", 0, quantize_row_q2_0_ref, dequantize_row_q2_0);
    emit_type<block_q3_0, QK3_0>("q3_0", 1, quantize_row_q3_0_ref, dequantize_row_q3_0);
    emit_type<block_q3_0_64, QK3_0_64>("q3_0_64", 1, quantize_row_q3_0_64_ref, dequantize_row_q3_0_64);
    return 0;
}
