#include "common.cuh"
#include "lfm2-shortconv.cuh"

// Fused LFM2.5 shortconv block:
//   u     = b * x                                  (elementwise, {n_embd, n_t, n_seqs})
//   v     = [conv_state ; u]                       (concat along ne[0], {D+n_t, n_embd, n_seqs})
//   o_t   = sum_j conv_kernel[j, ch] * v[t+j]      (causal conv1d, K taps, K = D+1)
//   dst   = c * o                                  (elementwise)
//   state = v[n_t .. n_t+D-1]                      (new state, written in-place to conv_state_dst)
//
// Replaces mul(b,x) + concat + ssm_conv + cpy(state) + mul(c,o) with a single kernel:
// 4 launches saved per shortconv layer per step.

// timesteps processed per block; conv causality is kept by seeding each
// block's u-ring with the D recomputed prior u values (cheap: D <= 14)
#define LFM2_SHORTCONV_BLOCK_T 32

template <int K>
static __global__ void lfm2_shortconv_f32_kernel(
        const float * __restrict__ b,      const int b_nb1,  const int b_nb2,
        const float * __restrict__ x,      const int x_nb1,  const int x_nb2,
        const float * __restrict__ c,      const int c_nb1,  const int c_nb2,
        const float * __restrict__ st_r,   const int st_nb1, const int st_nb2,
        const float * __restrict__ w,      const int w_nb1,
        float * __restrict__ st_w,
        float * __restrict__ dst,          const int dst_nb1, const int dst_nb2,
        const int64_t n_embd, const int64_t n_t) {
    constexpr int D = K - 1;  // state rows
    constexpr int TPB = 128;  // threads per block == channels per chunk
    const int chunk  = blockIdx.x;
    const int s      = blockIdx.y;
    const int64_t tBegin = (int64_t) blockIdx.z * LFM2_SHORTCONV_BLOCK_T;
    const int64_t tEnd   = min(tBegin + LFM2_SHORTCONV_BLOCK_T, n_t);

    const int tid = threadIdx.x;
    const int64_t st_nb0 = (int64_t) sizeof(float);
    const int64_t ch = (int64_t) chunk * TPB + tid;  // n_embd % 128 == 0 asserted

    float wv[K];
#pragma unroll
    for (int j = 0; j < K; j++) {
        wv[j] = w[ch * (w_nb1 / (int) sizeof(float)) + j];
    }

    // ring of the last u values; the conv window v[t .. t+K-1] needs
    // u_{t-D} .. u_t (D+1 = K distinct values), so the ring must be K wide.
    // Seed the D priors from the previous t-block, recomputed from b*x so the
    // blocks stay independent (at tBegin == 0, e.g. decode, the seed loop is a
    // no-op). Slot (idx % K) always holds u_idx.
    float u_ring[K] = { 0.0f };
    for (int64_t j = 0; j < (int64_t) D; j++) {
        const int64_t idx = tBegin - D + j;
        if (idx >= 0) {
            const float * b_pt = (const float *) ((const char *) b + ch * st_nb0 + idx * b_nb1 + s * b_nb2);
            const float * x_pt = (const float *) ((const char *) x + ch * st_nb0 + idx * x_nb1 + s * x_nb2);
            u_ring[idx % K] = (*b_pt) * (*x_pt);
        }
    }

    for (int64_t t = tBegin; t < tEnd; t++) {
        const float * b_pt = (const float *) ((const char *) b + ch * st_nb0 + t * b_nb1 + s * b_nb2);
        const float * x_pt = (const float *) ((const char *) x + ch * st_nb0 + t * x_nb1 + s * x_nb2);
        const float * c_pt = (const float *) ((const char *) c + ch * st_nb0 + t * c_nb1 + s * c_nb2);

        const float ut = (*b_pt) * (*x_pt);
        u_ring[t % K] = ut;

        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < K; j++) {
            const int64_t vi = t + j;
            float vj;
            if (vi < D) {
                // element (vi, ch, s) at vi*nb0 + ch*nb1 + s*nb2 (channel-blocked)
                vj = *(const float *) ((const char *) st_r + vi * st_nb0 + ch * st_nb1 + s * st_nb2);
            } else {
                vj = u_ring[(vi - D) % K];
            }
            acc += vj * wv[j];
        }

        *(float *) ((char *) dst + ch * st_nb0 + t * dst_nb1 + s * dst_nb2) = (*c_pt) * acc;
    }

    // the last t-block writes the new state: v[n_t .. n_t+D-1], channel-blocked
    // layout: element (j, ch) of seq s at (s*n_embd + ch)*D + j (matches the
    // cpy/cat layout). For vi < D this is an old-state row (state shift); the
    // remaining rows are the ring's last D u values.
    if (tEnd == n_t) {
#pragma unroll
        for (int j = 0; j < D; j++) {
            const int64_t vi = n_t + j;
            float sv;
            if (vi < D) {
                // element (vi, ch, s) at vi*nb0 + ch*nb1 + s*nb2 (channel-blocked)
                sv = *(const float *) ((const char *) st_r + vi * st_nb0 + ch * st_nb1 + s * st_nb2);
            } else {
                sv = u_ring[(vi - D) % K];
            }
            st_w[((int64_t) s * n_embd + ch) * D + j] = sv;
        }
    }
}

static void lfm2_shortconv_f32_cuda(
        const float * b,       const int b_nb1,  const int b_nb2,
        const float * x,       const int x_nb1,  const int x_nb2,
        const float * c,       const int c_nb1,  const int c_nb2,
        const float * st_r,    const int st_nb1, const int st_nb2,
        const float * w,       const int w_nb1,
        float * st_w,
        float * dst,           const int dst_nb1, const int dst_nb2,
        const int64_t K, const int64_t n_embd, const int64_t n_t, const int64_t n_seqs,
        cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(n_embd % threads == 0);
    const int64_t n_chunks = n_embd / threads;
    const int64_t n_t_blocks = (n_t + LFM2_SHORTCONV_BLOCK_T - 1) / LFM2_SHORTCONV_BLOCK_T;
    GGML_ASSERT(n_seqs <= 65535 && n_t_blocks <= 65535);
    const dim3 blocks((unsigned int) n_chunks, (unsigned int) n_seqs, (unsigned int) n_t_blocks);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        ggml_cuda_kernel_launch(lfm2_shortconv_f32_kernel<kNC>, launch_params,
                b, b_nb1, b_nb2, x, x_nb1, x_nb2, c, c_nb1, c_nb2,
                st_r, st_nb1, st_nb2, w, w_nb1, st_w, dst, dst_nb1, dst_nb2, n_embd, n_t);
    };

    switch (K) {
        case 2:  launch_kernel(std::integral_constant<int, 2 >{}); break;
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 2, 3, 4, 5, 9, 15 right now.");
    }
}

void ggml_cuda_op_lfm2_shortconv(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * b    = dst->src[0];
    const ggml_tensor * x    = dst->src[1];
    const ggml_tensor * c    = dst->src[2];
    const ggml_tensor * st_r = dst->src[3];
    const ggml_tensor * w    = dst->src[4];
    const ggml_tensor * st_w = dst->src[5];

    const int64_t n_embd = b->ne[0];
    const int64_t n_t    = b->ne[1];
    const int64_t n_seqs = b->ne[2];
    const int64_t K      = w->ne[0];  // conv kernel taps (l_cache)
    const int64_t D      = st_r->ne[0];  // state rows (l_cache - 1)

    GGML_ASSERT(b->type    == GGML_TYPE_F32);
    GGML_ASSERT(x->type    == GGML_TYPE_F32);
    GGML_ASSERT(c->type    == GGML_TYPE_F32);
    GGML_ASSERT(st_r->type == GGML_TYPE_F32);
    GGML_ASSERT(w->type    == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(st_w->type == GGML_TYPE_F32);
    GGML_ASSERT(b->nb[0] == sizeof(float));
    GGML_ASSERT(x->nb[0] == sizeof(float));
    GGML_ASSERT(c->nb[0] == sizeof(float));
    GGML_ASSERT(st_r->nb[0] == sizeof(float));
    GGML_ASSERT(w->nb[0]  == sizeof(float));
    GGML_ASSERT(K == D + 1);
    GGML_ASSERT(n_embd % 128 == 0);

    lfm2_shortconv_f32_cuda(
            (const float *) b->data,    (int) b->nb[1],    (int) b->nb[2],
            (const float *) x->data,    (int) x->nb[1],    (int) x->nb[2],
            (const float *) c->data,    (int) c->nb[1],    (int) c->nb[2],
            (const float *) st_r->data, (int) st_r->nb[1], (int) st_r->nb[2],
            (const float *) w->data,    (int) w->nb[1],
            (float *) st_w->data,
            (float *) dst->data,        (int) dst->nb[1],  (int) dst->nb[2],
            K, n_embd, n_t, n_seqs, ctx.stream());
}