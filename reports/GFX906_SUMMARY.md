# gfx906 Ryzen Tuning Fork - Final Summary

Scope: all gfx906 (Vega20 / GCN5, Radeon VII) tuning work done on the
`gfx906-ryzen-v2` branch, plus the generic host-side decode investigation on
the separate `host-decode-opt` branch. Written at branch HEAD
`d6ee7e7de` (gfx906-ryzen-v2).

## 1. Hardware and software

- CPU: Ryzen 7 3700X, 8C/16T, 32GB DDR4
- GPU: AMD Vega20 (gfx906), 60 CU, 16GB HBM2 (~1 TB/s), [1002:66af]
- Second GPU: NVIDIA GTX 1080 Ti (not used in this work)
- OS: Arch Linux, kernel 7.1.x
- ROCm: 7.2.4 (hipcc 7.2.53211), rocm-smi present
- clang 22.1.8, cuda 12.5 present but unused
- Models: Qwen3.5-4B (Q4_K_M primary, Q4_0, Q3_K_M, IQ4_XS), Qwen3-8B Q4_K_M

Build (ROCm 7.2 link workaround required):
```
cmake -B build-hip -DGGML_HIP=ON -DHIP_PATH=/opt/rocm \
      -DCMAKE_HIP_FLAGS="-I/opt/rocm/include"
cmake --build build-hip -j12
# link fix: sudo ln -sf /opt/rocm/lib/libamdhip64.so /opt/rocm/lib/lib/
```

## 2. Safe tower (committed wins)

Each commit is gfx906-gated (VEGA20 dispatch) unless noted.

| Commit | Change | Effect (Qwen3.5-4B Q4_K_M) |
|--------|--------|----------------------------|
| 32aafef50 | Q8_0 MMQ nthreads 256->512, J up to 128 with ncols<=128 clamp | +44% pp |
| ed41aba4e | VDR_Q4_K_MMVQ 2->8 (decode dequant) | +4.2% tg |
| 395f2d194 | FA vec nthreads 128->256 (VEGA20) | +3-7% tg |
| 200e23567 | FA tile 81 gfx906 configs | neutral (kept) |
| f70fd7f2d | Q4_K MMQ nthreads 256->512 | +9% pp |
| d5b9855c2 | Q5_K/Q6_K MMQ I 128->64 | +9.8% pp |
| 1ccb8bbbd | K_vram 256->512 (global MMQ_ITER_K) | +43% pp |
| 35b53a5f0 | FA tile KV nbatch_K 32->64 (128,128,64) | +6.5% pp |
| 5150bb02b | Stream-K true for all Q8_0 MMQ cases | +1-6% pp |
| f44476056 | op-timing harness + tower docs | diagnostic |
| d6ee7e7de | gate VDR_Q4_K_MMVQ behind VEGA20 | clean scoping |

Final benchmark (re-verified at free-boost clocks, stable, no SMU crash):
- Qwen3.5-4B Q4_K_M: pp512 1776->1781, pp8192 1686->1679, tg 83->92.7
- Qwen3.5-4B Q4_0: pp512 ~1380->2187
- Qwen3.5-4B Q3_K_M: pp512 1058->1716 (+62%)
- Qwen3.5-4B IQ4_XS: pp512 1148->1849 (+61%)
- Qwen3-8B Q4_K_M: pp512 803->1280 (+59%), tg 66.7->87.6 (+31%)

So prefill gains +59% to +73% across every quant family and dense 8B.
Decode (tg) gains scale with model size: flat on small 4B (host-bound),
+31% on 8B (GPU-bound decode).

## 3. Levers rejected for crash / correctness

| Lever | Result | Why rejected |
|-------|--------|--------------|
| K_vram 1024 | fake 4535 t/s, garbage output | silent K-tail skip (correctness bug) + GPU CPC_UCODE microcode corruption |
| Q4_K Stream-K true | abort in mul_mat_q_switch_J | code bug, not safe |
| KV q4_0 FA vec path | GPU fell off PCIe bus (SMU timeout 0x10) | untested wave64 + KV64 tile, hardware drop |
| IQ4_NL KV cache | -78% pp, timeout | not FA-whitelisted, falls to slow generic dequant |

## 4. Levers rejected (safe but no win)

| Lever | Result |
|-------|--------|
| KV q8_0 cache | -0.7% pp, -3.3% tg; only halves KV (2x slots), no speed |
| FA tile 128,128,32 nbatch_K 64->128 | -12% |
| FA tile exhaustive (8 more configs) | neutral (+0.2-0.8%) |
| compiler flags (-ffast-math, flush-denormals, warp-sync) | neutral (<0.5%) |
| speculative decoding (--speculate 5/8, 0.8B draft) | -1.4% (host launch-bound) |
| ubatch 1..512 | flat 91 t/s (per-token CPU gap) |
| Q5_K/Q6_K J 128 | regress (picker chose slow config) |
| 64k context | pp512->64k drops -46% (O(n^2) FA compute, not KV BW) |

## 5. Decode latency root cause (generic, host-side)

A gated CPU timer (GGML_DECODE_CPU_TIMING, on host-decode-opt branch)
split the per-token cost:

- decode (llama_decode) CPU: ~0.4 ms/tok
- sample CPU total: ~12.5 ms/tok
  - of which llama_synchronize (ROCm sync): ~12.0 ms/tok
  - CPU sampler logic (argmax/grammar): ~0.5 ms/tok

Disabling graphs (GGML_CUDA_DISABLE_GRAPHS=1): 11 ms (no improvement).
0.8B model: sync ~5.5 ms/tok (scales with model, GPU compute only ~0.1 ms).

Conclusion: single-stream decode tg on gfx906/ROCm7.2 is bounded by
ROCm per-token host->GPU submit/synchronize LATENCY. The GPU is idle
97-98% per token. Graph capture is active but does not reduce it. No
kernel or CPU-sampler change can raise single-stream tg. This is generic
(all backends), not gfx906-specific.

## 6. Practical tg win: parallel batching

llama-batched-bench aggregate (single GPU, f16 KV):

- 4B Q4_K_M: B1 90 -> B2 134 (1.49x) -> B4 164 (1.82x) -> B8 190 t/s (2.11x)
- 8B Q4_K_M: B8 204 t/s (2.31x)

Prefill stays flat (94% util). The 10ms host gap is filled by batching.
This is a run mode (llama-batched-bench -npl N, llama-server --parallel N),
not a code change. q8_0 KV doubles slot count for multi-user but costs
~3% tg, so f16 is the low-latency default and q8_0 optional for capacity.

## 7. GPU stability (Vega20 SMU/DPM)

Three PC crashes (m0579, m0595, m0673) had signature
`amdgpu ... Failed to send message 0x10` + athub errors. Two causes:
- risky lever tests (KV q4_0 FA vec path, perf record interrupt load)
- a known Vega20 SMU/DPM firmware bug triggered by DPM clock transitions
  under sustained compute load (not the tuning code itself)

Resolution: clocks are managed via LACT at free-boost (sclk up to 1700 MHz,
mclk 1000 MHz), fixed (no DPM transitions). This gives full tower speed
with no SMU crash. Earlier `amdgpu.dpm=0` grub hack locked the GPU to a low
boot clock (327 t/s, 5x slower) and was reverted; it is not the fix.
Prefill is memory-bandwidth bound, so mclk (vram) at 1000 MHz matters most;
core 700 MHz is enough once memory is unstarved.

## 8. Branch layout

- gfx906-ryzen-v2 (pushed to PetrouilFan/llama.cpp, no PR): the gfx906-only
  GPU-tuning fork. All MMQ/FA/VDR changes are gfx906-gated. Keep it this way.
- host-decode-opt (local, from master 511f9c137): generic host-side decode
  diagnostic timer (uncommitted, gated, no behavior change when env off).
  Do NOT merge generic changes into gfx906-ryzen-v2.

## 9. Final verdict

The gfx906 kernel space is fully explored. Single-stream prefill is +60-73%
via memory-trip reductions (K512, Stream-K, 512thr, FA KV64). Single-stream
decode is capped by ROCm launch latency (~90 t/s 4B) and only parallel
batching raises aggregate tg (2.11x). No further safe single-stream win
exists on this hardware. The fork is complete and stable.
