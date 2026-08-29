# gfx906 (Vega20 / Radeon VII) HIP Tuning Tower

Personal optimization branch `gfx906-ryzen-v2`. Not for upstream.

## Hardware target

- GPU: AMD Radeon VII, `gfx906:sramecc+:xnack-` (Vega20 / GCN5), 60 CUs, 16 GiB HBM2 (~1 TB/s), wave64.
- Host: Ryzen 7 3700X, 32 GiB DDR4 (no CPU compute bottleneck).
- Stack: ROCm 7.2.4 (`rocm-hip-sdk`), clang/LLVM from `/opt/rocm`.

Key property: HBM2 at ~1 TB/s means the GPU is **bandwidth-rich, compute-poor** and (as measured) **launch/dispatch-bound in decode**. This shapes every tuning choice below.

## Baseline

- Vanilla ancestor: `511f9c137` (master) built with the HIP backend.
- Build flags (required on this ROCm):

```
cmake -B build-hip -S . \
  -DGGML_HIP=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DHIP_PATH=/opt/rocm \
  -DCMAKE_HIP_FLAGS="-I/opt/rocm/include" \
  -DGGML_HIP_GRAPHS=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON
```

(ROCm 7.2 also needs `/opt/rocm/lib/lib` symlinked to `libamdhip64.so` for the linker.)

## The tuning layer (portable, by construction generic)

All changes apply to every model/quant, not one architecture quirk:

1. `vendors/hip.h` - add `GCN5`/`VEGA20` arch defs (`GGML_CUDA_CC_VEGA20`).
2. `mmq.cuh` - add `VEGA20` device+host dispatch (gfx906 previously fell through to the `RDNA2` table) + clamp `J<=64` when `ncols<=128`.
3. `mmq-config-gfx906.cuh` - dedicated config table: `Q8_0`/`Q4_K` `nthreads 512`, `Q5_K`/`Q6_K` `I 64`, `Stream-K true` for `Q8_0`.
4. `mmq.cuh` - `MMQ_ITER_K 256 -> 512` (`K_vram`): each block does 2x K before a global store. **The single biggest win** (halves stores; fits because HBM hides the larger LDS working set).
5. `fattn-tile.cuh` - dedicated gfx906 tile config + `KV tile 64` (`nbatch_K 32->64`) at the `128,128,64` case.
6. `fattn-vec.cuh` - FA vec `nthreads 256` (4x wave64).
7. `vecdotq.cuh` - `VDR 2 -> 8` (decode dot product).

## Results (tokens/s, Radeon VII)

Tower = `gfx906-ryzen-v2` at `5150bb02b`. Vanilla = `511f9c137`.

| model (quant)            | pp512  (van->tower) | pp1024 | pp8192 | tg128 |
|--------------------------|---------------------|--------|--------|-------|
| Qwen3.5-4B Q4_K_M        | 1024 -> 1776 (+73%) | 1071->1811 | 1105->1686 | 83->91 (+10%) |
| Qwen3.5-4B Q4_0          | ~1380 -> 2197 (+59%) | - | - | 93->93 (flat) |
| Qwen3.5-4B Q3_K_M        | 1058 -> 1716 (+62%) | 1061->1771 (+67%) | 988->1650 (+67%) | 74->80 (+8%) |
| Qwen3.5-4B IQ4_XS        | 1148 -> 1849 (+61%) | 1148->1903 (+66%) | 1068->1765 (+65%) | 91->92 (+0.6%) |
| Qwen3-8B Q4_K_M (dense)  | 803 -> 1280 (+59%) | 789->1262 (+60%) | 638->888 (+39%) | 66.7->87.6 (+31%) |

Takeaways:
- **Prefill: +60-73% across every quant family** (plain Q4_0, mixed K_M, low-bit Q3_K/IQ4_XS, dense 8B). The `K512` + `Stream-K` + `512thr` memory-trip reductions hit the universal `MUL_MAT` path.
- **tg gain scales with model size**: small 4B models are host/dispatch-bound in decode (GPU idle ~97%), so tg is ~flat; larger 8B is GPU-bound in decode, so tg gains +31%. This is structural, not quant-specific.

## Out-of-box verification (overnight, all safe, all generic)

After the kernel tower, every non-kernel lever was measured to confirm the ceiling:

- **KV cache quant**: `f16` baseline kept. `q8_0` safe but `pp -2%`/`tg -3%` (dp4a dequant offsets BW save); only useful as **capacity** (halves KV `1GB->0.5GB` at `c8192`, `4B 13->26` slots). `q4_0` FA vec path **crashed the GPU** (SMU/PCIe drop) -> rejected. `IQ4_NL` not FA-whitelisted -> `-78% pp` + timeout -> rejected. Verdict: `f16` for latency, `q8_0` optional for 4B high-throughput.
- **64k realistic**: `pp512->64k` drops `-46%` (O(n^2) FA compute, not KV BW); `q8_0` still `-2%` even at `pp65408`. KV quant does not fix large context.
- **Compiler flags**: `-ffast-math`, `-fgpu-flush-denormals-to-zero`, `HIP_ENABLE_WARP_SYNC` -> all neutral (`<0.5%`).
- **FA tile exhaustive**: 8 more `nbatch_K`/`nbatch_fa` candidates (40..512 heads, LDS<=29KB) -> all neutral (`+0.2-0.8%`). `128,128,64 32->64` (`+6.5%`) remains the only win.
- **Decoder batch (`-ub` 1..512)**: flat `91 t/s` -> the 10ms gap is per-token CPU, not amortizable by ubatch.
- **Speculative decoding** (`--speculate 5/8`, 0.8B draft): `-1.4%` -> no win because bottleneck is host launch latency, not GPU compute.
- **Parallel slots** (`llama-batched-bench`, single-stream aggregate): `B8 S_TG 1.78-2.11x` at `94%` prefill util; 8B `2.31x`. This is the practical tg lever for throughput.
- **MMVQ**: no `VEGA20`/`gfx906` arch path exists; decode matmul is fully generic. Added `nwarps 4` earlier -> regress, reverted. Exhausted.

Conclusion: the gfx906 **kernel** space is fully explored. Single-stream decode tg is bounded by per-token host CPU (~10ms) which no GPU change can remove.

## Methodology (why these and not others)

An op-timing harness (`GGML_CUDA_OP_TIMING`, see `ggml-cuda.cu`) measured per-op GPU time:
- Prefill: `MUL_MAT` ~77% of GPU time at **94% GPU utilization** -> matmul-bound -> the `K512`/`Stream-K`/`512thr` tower is the right target, and it is exhausted (no more memory-trip headroom).
- Decode: GPU busy only **~3%** of wall time. The decode CUDA graph replays each token (one `cudaGraphLaunch`, microseconds); the ~10 ms gap is host-side (`llama.cpp` sampling + scheduler + ubatch prep). Therefore **no gfx906 kernel change can raise tg** on small models. Only host-side llama.cpp features (speculative decoding, continuous batching, GPU-side sampling) move tg, and those are generic across backends.

## Rejected / neutral (do not re-test)

All measured on this hardware, recorded in `reports/EXPERIMENT_LEDGER.md`:
- `Q3_K`/`Q2_K`/`Q6_K`/`MXFP4` `nthreads 256->512` - neutral/regress (VGPR-bound, not bandwidth).
- `vecdotq` `VDR 8->16` - plateau.
- `Q4_K` `J 64->128` - mixed, regress at pp512.
- `gated_delta_net` 4->8 warps, occupancy 1 - flat (`GATED_DELTA_NET` is not a standalone op; DeltaNet uses generic `MUL_MAT`/`CONCAT`/`GET_ROWS` already covered).
- `fattn` `KV tile 128` - LDS infeasible (64 KiB limit).
- `k_get_rows` - already `int4` (128-bit) loads; random row gather not coalescable.

## Rebuild

```
rm -rf build-hip/ggml/src/ggml-hip          # force recompile of mmq/fattn after config edits
cmake --build build-hip -j12
```

## Status

The gfx906 **kernel** tuning is complete and near-optimal. Remaining levers are host-side llama.cpp features (speculative decoding etc.), which are out of scope for a backend-specific tower.
