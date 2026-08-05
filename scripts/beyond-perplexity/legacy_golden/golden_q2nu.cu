// q2_nu golden generator: runs the legacy device quantize/dequant code paths
// (q2-nu-codebook.cuh + quantize_f32_q2_0_block from cpy-utils.cuh) on fixed
// inputs and emits one JSON record per case, mirroring golden_host format.
//
// Usage (from the frozen legacy repo root):
//   nvcc -O2 -std=c++17 \
//       -I $LEGACY/ggml/include -I $LEGACY/ggml/src -I $LEGACY/ggml/src/ggml-cuda \
//       golden_q2nu.cu -o golden_q2nu
//   ./golden_q2nu > golden_q2nu.raw
//
// Output format (one JSON object per line):
//   {"type":"q2_nu","case":"thresh","sub":2,"input":[...hex floats...],
//    "scale":"0x0000","packed":"...hex...","dequant":[...hex floats...]}
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>

#include "ggml-common.h"
#include "q2-nu-codebook.cuh"
#include "cpy-utils.cuh"

constexpr int N = 64;

static __global__ void quant_kernel(const float * x, block_q2_0 * y) {
    quantize_f32_q2_0_block(x, y);
}

static __global__ void dequant_kernel(const block_q2_0 * y, float * out) {
    const int j = threadIdx.x;
    const float d = __half2float(y->d);
    const int byte_idx = j / 4;
    const int bit_off = (j % 4) * 2;
    const int code = (y->qs[byte_idx] >> bit_off) & 3;
    out[j] = q2_nu_dequant(code, d);
}

static void hexf(float v, char * buf) {
    uint32_t u;
    memcpy(&u, &v, 4);
    snprintf(buf, 16, "%08x", u);
}

int main() {
    const char * cases[8] = {"zeros", "const_pos", "const_neg", "extrema", "thresh", "all_codes", "all_positions", "random"};
    const char * subs[4] = {"a", "b", "c", "d"};
    const float eps[4] = {-1e-5f, 1e-5f, -1e-3f, 1e-3f};
    const float lv[4] = {-0.619326f, -0.186363f, 0.186363f, 0.619301f};
    for (int c = 0; c < 8; ++c) {
        for (int s = 0; s < 4; ++s) {
            std::vector<float> x(N, 0.0f);
            switch (c) {
                case 0: break; // zeros: degenerate d <= 0 path
                case 1: for (int j = 0; j < N; ++j) x[j] = 0.75f; break;
                case 2: for (int j = 0; j < N; ++j) x[j] = -0.75f; break;
                case 3: {
                    const float ext[4] = {0.001f, 3.0f, -2.5f, 1e-4f};
                    for (int j = 0; j < N; ++j) x[j] = ext[(j + s) % 4]; break;
                }
                case 4: { // all three boundaries at once; d = 1 via x[0]
                    x[0] = 1.0f;
                    x[1] = -0.403f + eps[s];
                    x[2] =  0.0f   + eps[s];
                    x[3] =  0.403f + eps[s]; break;
                }
                case 5: { // all 4 codes at exact level values, d = 1
                    x[0] = 1.0f;
                    for (int j = 1; j < N; ++j) x[j] = lv[(j - 1) % 4]; break;
                }
                case 6: { // every packed position, cycling all codes
                    x[0] = 1.0f;
                    for (int j = 1; j < N; ++j) x[j] = lv[((j - 1) * 7 + s) % 4]; break;
                }
                case 7: { // random, fixed seed
                    srand(12345 + s * 7919);
                    for (int j = 0; j < N; ++j) {
                        x[j] = (float) (rand() % 20001) / 10000.0f - 1.0f;
                    }
                } break;
            }
            float * dx; block_q2_0 * dy; float * dout;
            cudaMalloc(&dx, N * sizeof(float));
            cudaMalloc(&dy, sizeof(block_q2_0));
            cudaMalloc(&dout, N * sizeof(float));
            cudaMemcpy(dx, x.data(), N * sizeof(float), cudaMemcpyHostToDevice);
            quant_kernel<<<1, 1>>>(dx, dy);
            dequant_kernel<<<1, N>>>(dy, dout);
            std::vector<float> out(N);
            cudaMemcpy(out.data(), dout, N * sizeof(float), cudaMemcpyDeviceToHost);
            block_q2_0 y;
            cudaMemcpy(&y, dy, sizeof(block_q2_0), cudaMemcpyDeviceToHost);

            char buf[32];
            printf("{\"type\":\"q2_nu\",\"case\":\"%s\",\"sub\":%d,\"input\":[", cases[c], s);
            for (int j = 0; j < N; ++j) { hexf(x[j], buf); printf("{\"hex\":\"%s\",\"f\":\"%.9g\"}", buf, (double)x[j]); if (j < N-1) printf(","); }
            printf("],\"scale\":\"%04x\",\"packed\":\"", (unsigned) __half_as_ushort(y.d));
            for (int j = 0; j < 16; ++j) printf("%02x", (unsigned) y.qs[j]);
            printf("\",\"dequant\":[");
            for (int j = 0; j < N; ++j) { hexf(out[j], buf); printf("{\"hex\":\"%s\",\"f\":\"%.9g\"}", buf, (double)out[j]); if (j < N-1) printf(","); }
            printf("]}\n");
            cudaFree(dx); cudaFree(dy); cudaFree(dout);
        }
    }
    return 0;
}
