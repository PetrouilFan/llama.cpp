# Q3 copy-path fix (cpy_blck_q_f32_pair)

Date: 2026-08-05

## 1. Summary

The CUDA copy path for quantized-to-F32 (`cpy_q_f32`) was broken for the
new element-consecutive dequantize kernels (Q2_NU, Q3_0, Q3_0_64). A new
template `cpy_blck_q_f32_pair` was added to cpy.cu and the three dispatch
entries now use it. Without the fix, `CPY type_src=<new type> type_dst=f32`
produced fully wrong output (error ~1.0 in test-backend-ops).

## 2. Exact legacy bug

The dequantize kernels in dequantize.cuh follow two conventions depending on
`qr`:

- qr=2 types (Q4_0, Q4_1, Q5_0, Q5_1): the caller passes the quant index
  `iqs` in [0, qk/2) and the kernel returns `v.x = element iqs`,
  `v.y = element iqs + qk/2`. The generic `cpy_blck_q_f32` template writes
  `cdstf[j]`, `cdstf[j + qk/2]` for `j` in [0, qk/2), which matches.
- qr=1 types (Q2_0 and the new Q2_NU, Q3_0, Q3_0_64): the caller passes the
  even element index `iqs` in {0, 2, ...} and the kernel returns two
  consecutive elements `v.x = element iqs`, `v.y = element iqs + 1`.

The legacy `cpy_blck_q_f32` template was applied unchanged to the new types,
so it wrote element `iqs` to `cdstf[j]` and `cdstf[j + qk/2]` with `qk/2`
wrongly treated as the pair stride. The same defect exists in the legacy
archive: legacy cpy.cu routes q3_0->f32 through the identical generic
template, so legacy GPU output for this path was wrong as well. The legacy
project never ran this path on GPU (its validation was CPU-only golden
vectors).

## 3. Exact affected types

- `GGML_TYPE_Q2_NU` (QK2_NU = 64)
- `GGML_TYPE_Q3_0` (QK3_0 = 32)
- `GGML_TYPE_Q3_0_64` (QK3_0_64 = 64)

The qr=2 types and the scalar (F16/BF16/F32) paths were not affected.

## 4. Fix

Added to ggml/src/ggml-cuda/cpy.cu (template immediately after
`cpy_blck_q_f32`):

```cpp
// for types where the dequant pair is consecutive (iqs, iqs + 1)
template<dequantize_kernel_t dequant, int qk>
static __device__ void cpy_blck_q_f32_pair(const char * cxi, char * cdsti) {
    float * cdstf = (float *)(cdsti);
#pragma unroll
    for (int j = 0; j < qk; j += 2) {
        float2 dq;
        dequant(cxi, 0, j, dq);
        *(cdstf + j) = dq.x;
        *(cdstf + j + 1) = dq.y;
    }
}
```

This mirrors the existing `cpy_blck_q8_0_f32` pattern. The three dispatch
entries now read:

- `cpy_q_f32<cpy_blck_q_f32_pair<dequantize_q2_nu, QK2_NU>, QK2_NU>`
- `cpy_q_f32<cpy_blck_q_f32_pair<dequantize_q3_0, QK3_0>, QK3_0>`
- `cpy_q_f32<cpy_blck_q_f32_pair<dequantize_q3_0_64, QK3_0_64>, QK3_0_64>`

## 5. Old versus new golden output

test-backend-ops (backend CUDA0) `CPY` cases:

- Before: 6 failures, all of the form
  `CPY(type_src=q2_nu|q3_0|q3_0_64, type_dst=f32, ne_src=[256,4,4,4])` and the
  `[256,2,3,4]` permuted variant, error ~1.0 (values completely wrong).
- After: all 6 pass with error 0.0; the filtered matrix
  (`-p "q3_0|q3_0_64|q2_nu"`) is 102/102 OK, full CUDA0 suite 13096/13096 OK.

## 6. Did the historical paper path use this path?

No. The paper path (FlashAttention) never copies quantized cache blocks to
F32. CPY into the cache comes from host F32 scratch (supported both before
and after), and quantized-to-F32 CPY is only reachable through non-FA
attention or explicit data movement, which the paper runs did not use.
The fix only affects the CUDA backend correctness of a path the paper did
not exercise; no historical value changes.

## 7. Sanitizer/backend evidence

- test-backend-ops CUDA0 full suite after the fix: 13096/13096 OK, 0 failed.
- CPU reference in ggml-quants.c (`dequantize_row_q2_nu`,
  `dequantize_row_q3_0`, `dequantize_row_q3_0_64`) is element-indexed; the
  fixed kernels produce identical element ordering.
- No sanitizer issues observed on the CPY path (see COMPUTE_SANITIZER_REPORT.md
  for the real-run sanitizer gate).