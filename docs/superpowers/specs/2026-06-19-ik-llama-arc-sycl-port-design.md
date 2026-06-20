# ik_llama.cpp Intel Arc SYCL Port Design

## Goal

Port the useful Intel Arc / SYCL backend improvements from latest `llama.cpp`
into latest `ik_llama.cpp` so the real inference host can run `ik_llama.cpp`
effectively on an Intel Arc B580 with the i9 CPU and 64 GB RAM.

The coding container is not the runtime target. It has limited CPU visibility
and does not expose the Arc GPU. Work in this container should focus on source
analysis, conflict resolution, and build setup. Runtime validation must happen
on the inference host.

## Current Context

`llama.cpp` has an actively maintained SYCL backend for Intel GPUs. Its latest
tree has SYCL code split under `ggml/src/ggml-sycl/`, with recent changes around
Arc B-Series support, oneAPI 2025.x support, Level Zero allocation, optional
USM/system allocation behavior, device-to-device copies, FP16 defaults, and new
operation coverage.

`ik_llama.cpp` keeps SYCL in a single `ggml/src/ggml-sycl.cpp` file and carries
many independent performance and runtime changes. The local branch
`fix-sycl-buffer-interface` contains targeted compatibility fixes:

- repair the SYCL backend buffer interface after upstream API drift
- update SYCL tensor code to use `__backend`
- mark normal SYCL tensors as `GGML_BACKEND_TYPE_GPU`
- skip speculative probing when speculative decoding is unused
- disable split mode graph for recurrent/hybrid models with tensor overrides

That branch is behind latest `origin/main`, so the first task is to rebase or
reconstruct the branch on top of current `ik_llama.cpp`.

## Non-Goals

Do not wholesale replace `ik_llama.cpp`'s ggml backend with `llama.cpp`'s
backend. The repos have diverged, and `ik_llama.cpp` includes model-specific and
runtime-specific optimizations that should not be discarded.

Do not optimize for CUDA, Vulkan, Metal, or AMD. This pass is specifically for
Intel Arc B580 through SYCL / Level Zero.

Do not rely on the coding container for performance conclusions. It is useful
for code work, not for Arc runtime validation.

## Approach

### 1. Rebase The Working Branch

Bring the SYCL branch onto latest `ik_llama.cpp origin/main`. Prefer replaying
the small local commits over merging the old branch wholesale. During conflict
resolution, keep only changes that still apply to current `origin/main`.

Expected branch shape:

- latest `origin/main`
- minimal SYCL compatibility fixes still missing from main
- any small runtime fixes that are required for the target model/server flow

### 2. Compare SYCL Backend Deltas

Compare latest `llama.cpp` SYCL implementation against latest `ik_llama.cpp`.
Focus on behavior, not file layout. The main comparison areas are:

- buffer interface callbacks and tensor initialization
- `GGML_BACKEND_TYPE_GPU` and `GGML_BACKEND_TYPE_GPU_SPLIT` marking
- Level Zero allocation and large allocation behavior
- Arc B-Series / B580 support assumptions
- `GGML_SYCL_*` build options and defaults
- FP16 build behavior
- device-to-device memory copy behavior
- host memory fallback behavior
- operation coverage and performance kernels relevant to common quantized LLMs
- documentation and build command drift

### 3. Port Narrowly

Port compatibility and memory-management changes before performance kernels.
Each port should be small enough to review by behavior:

1. Build and backend option parity.
2. Tensor/backend state correctness.
3. Level Zero and large allocation handling.
4. Device selection and memory-copy behavior.
5. Arc-relevant performance kernels only after the backend is stable.

Avoid moving `ik_llama.cpp` from its single-file SYCL layout to `llama.cpp`'s
multi-file layout unless the port becomes unmaintainable without that split.

### 4. Validate In Layers

In the coding container:

- inspect diffs against both repos
- verify branch state is clean before and after changes
- run available source-level checks
- prepare exact build commands for the runtime host

On the Arc B580 inference host:

- confirm `sycl-ls` sees the B580 through Level Zero
- configure and build `ik_llama.cpp` with `GGML_SYCL=ON`
- run `llama-server --list-devices`
- run the target model with a representative prompt
- benchmark against latest `llama.cpp` on the same host
- capture prompt processing and token generation rates
- keep one known-good command line for production use

## Candidate Build Commands

Baseline Linux SYCL build for the inference host:

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build-sycl-b580 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_SYCL=ON \
  -DGGML_SYCL_F16=ON \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx
cmake --build build-sycl-b580 --target llama-server llama-cli llama-bench -j20
```

If FP16 is slower or unstable for the selected model, rebuild the same profile
with `-DGGML_SYCL_F16=OFF` and compare.

## Risks

`llama.cpp` and `ik_llama.cpp` have diverged enough that direct cherry-picks may
look clean but change runtime behavior incorrectly. Ports should be behavior-led
and reviewed against both source trees.

SYCL improvements in `llama.cpp` may depend on surrounding allocator or graph
changes that do not exist in `ik_llama.cpp`. In those cases, either port the
minimum dependency chain or skip the change.

The coding container cannot prove Arc correctness. A change is not complete
until it builds and runs on the B580 host.

## Success Criteria

- `ik_llama.cpp` has a current branch based on latest `origin/main`.
- The previous local SYCL compatibility fixes are either upstreamed, reapplied,
  or documented as obsolete.
- Arc/SYCL backend deltas from `llama.cpp` are inventoried and prioritized.
- The selected compatibility and memory-management improvements are ported with
  small, reviewable commits.
- The B580 host can build `ik_llama.cpp` with SYCL and list the Arc device.
- The target server flow runs on the B580 host.
- Benchmark results make clear whether the port is worth continuing into deeper
  performance-kernel work.
