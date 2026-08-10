# gfx906 Optimization Environment Record

Date/time : 2026-08-10 15:09 EEST
Hostname  : skatopc
OS        : EndeavourOS (Arch), kernel 7.1.5-arch1-2 x86_64
CPU       : Intel Core i5-3570 @ 3.40 GHz (4 cores/4 threads, Ivy Bridge)

## GPU

Card     : AMD Radeon VII (Vega 20, gfx906, GCN5)
GFX      : gfx906 (sramecc+, xnack-)
Compute Units : 60, SIMDs/CU 4, wavefront size 64
VRAM     : 16 GB HBM2 (17163091968 bytes total)
PCIe     : 8.0 GT/s x16
Idle power/temp at capture: 20 W, 37C edge / 39C junction, fclk 1225, sclk 808, mclk 350 MHz

## Software

ROCm   : 7.2.53211-9999 (HIP_VERSION 7.2.53211)
HIP    : /opt/rocm (HIP_PATH=/opt/rocm, HIP_PLATFORM=amd, runtime rocclr)
hipcc  : AMD clang version 22.0.0git (rocm-llvm f58b06dce1f9c15707c5f808fd002e18c2accf7e)
System clang : 22.1.8
rocblas: librocblas.so.5.2 present
hipblaslt present
Profiler tools: NONE installed (rocprof/rocprofv2/v3/rocprof-compute/omniperf all absent)

## Repository

Branch : gfx906-optimize (created fresh from master)
Commit : 62bf73d25 (model: Muse Glimmer Support (#26841))
Upstream : origin/master == 62bf73d25 (fetched, no divergence)
Worktree : clean before branch creation; single worktree at /home/petrouil/Projects/llama.cpp
Remote : https://github.com/PetrouilFan/llama.cpp.git

## Build Configuration (baseline)

cmake -S . -B build-gfx906 -DGGML_HIP=ON -DGPU_TARGETS=gfx906 -DCMAKE_BUILD_TYPE=Release

GPU_TARGETS forwards to CMAKE_HIP_ARCHITECTURES in ggml/src/ggml-hip/CMakeLists.txt.

## Models (benchmark target)

/home/petrouil/Projects/test_models/
  LFM2.5-2.6B-Q4_0.gguf   (1593894720 bytes)
  LFM2.5-2.6B-Q4_K_M.gguf (1674454848 bytes)
  LFM2.5-2.6B-Q8_0.gguf   (2874779456 bytes)