# ik_llama.cpp Intel Arc SYCL Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the smallest useful set of latest `llama.cpp` SYCL / Intel Arc improvements into current `ik_llama.cpp` so the B580 inference host can build, list, run, and benchmark `ik_llama.cpp` through SYCL.

**Architecture:** Work from a fresh branch based on latest `ik_llama.cpp origin/main`, then add changes in reviewable layers: branch reconstruction, delta inventory, CMake/backend option parity, tensor/backend correctness, Level Zero and host-memory fallback behavior, and host validation documentation. Keep `ik_llama.cpp`'s SYCL layout unless a specific upstream file must be copied to unlock a tested feature.

**Tech Stack:** CMake, Ninja, Intel oneAPI SYCL (`icx`/`icpx`), Level Zero, ggml SYCL backend, Git worktrees, Markdown validation notes.

---

## File Structure

- Modify: `ggml/CMakeLists.txt`
  - Adds missing SYCL options copied from latest `llama.cpp`: graph toggle, host memory fallback, Level Zero API toggle, oneDNN toggle, and SYCL device architecture cache variable.
- Modify: `ggml/src/CMakeLists.txt`
  - Wires the new SYCL options into the existing `ik_llama.cpp` SYCL target with minimal disruption.
- Modify: `ggml/src/ggml-sycl.cpp`
  - Keeps or reapplies tensor backend correctness and any top-level backend glue still required after rebasing.
- Modify: `ggml/src/ggml-sycl/*.cpp` and `ggml/src/ggml-sycl/*.hpp`
  - Ports only specific Arc-relevant behavior that exists in latest `llama.cpp` and is missing in `ik_llama.cpp`.
- Modify: `docs/backend/SYCL.md`
  - Updates Intel Arc B580 and oneAPI guidance to match the final build behavior.
- Create: `docs/superpowers/reports/2026-06-19-sycl-delta-inventory.md`
  - Records what was compared, what was ported, what was skipped, and why.
- Create: `docs/superpowers/reports/2026-06-19-b580-validation.md`
  - Gives exact commands for the real B580 inference host and a table for measured results.

## Task 1: Create A Fresh Port Branch

**Files:**
- No source files modified.

- [ ] **Step 1: Verify current branch is clean**

Run:

```bash
cd /root/git/ik_llama.cpp
git status --short --branch
git log --oneline -5
```

Expected:

```text
## fix-sycl-buffer-interface...sergio/fix-sycl-buffer-interface [ahead 1]
2b55cc55 docs: design Arc SYCL port
```

- [ ] **Step 2: Create a dedicated worktree from latest upstream main**

Run:

```bash
cd /root/git
git -C ik_llama.cpp fetch origin
git -C ik_llama.cpp worktree add /root/git/ik_llama-arc-sycl-port -b arc-sycl-port origin/main
```

Expected:

```text
Preparing worktree (new branch 'arc-sycl-port')
HEAD is now at d47f484d Force Gemma4 assistant to be loaded on last GPU (#1999)
```

- [ ] **Step 3: Copy the approved spec and this plan into the worktree**

Run:

```bash
cd /root/git/ik_llama-arc-sycl-port
mkdir -p docs/superpowers/specs docs/superpowers/plans
cp /root/git/ik_llama.cpp/docs/superpowers/specs/2026-06-19-ik-llama-arc-sycl-port-design.md docs/superpowers/specs/
cp /root/git/ik_llama.cpp/docs/superpowers/plans/2026-06-19-ik-llama-arc-sycl-port.md docs/superpowers/plans/
git add docs/superpowers/specs/2026-06-19-ik-llama-arc-sycl-port-design.md docs/superpowers/plans/2026-06-19-ik-llama-arc-sycl-port.md
git commit -m "docs: add Arc SYCL port plan"
```

Expected:

```text
[arc-sycl-port ...] docs: add Arc SYCL port plan
```

## Task 2: Inventory SYCL Delta Against llama.cpp

**Files:**
- Create: `docs/superpowers/reports/2026-06-19-sycl-delta-inventory.md`

- [ ] **Step 1: Create the report skeleton**

Create `docs/superpowers/reports/2026-06-19-sycl-delta-inventory.md` with:

```markdown
# SYCL Delta Inventory

## Inputs

- Source repo: `/root/git/llama.cpp`
- Target repo: `/root/git/ik_llama-arc-sycl-port`
- Runtime target: Intel Arc B580 through SYCL / Level Zero

## Ported

| Area | llama.cpp source | ik_llama.cpp target | Commit | Reason |
| --- | --- | --- | --- | --- |

## Skipped

| Area | llama.cpp source | Reason |
| --- | --- | --- |

## Requires B580 Host Validation

| Check | Command | Expected Evidence |
| --- | --- | --- |
```

- [ ] **Step 2: Generate file-list comparison**

Run:

```bash
cd /root/git
find llama.cpp/ggml/src/ggml-sycl -maxdepth 2 -type f | sed 's#^llama.cpp/ggml/src/ggml-sycl/##' | sort > /tmp/llama-sycl-files.txt
find ik_llama-arc-sycl-port/ggml/src/ggml-sycl -maxdepth 2 -type f | sed 's#^ik_llama-arc-sycl-port/ggml/src/ggml-sycl/##' | sort > /tmp/ik-sycl-files.txt
comm -23 /tmp/llama-sycl-files.txt /tmp/ik-sycl-files.txt > /tmp/sycl-files-only-in-llama.txt
comm -12 /tmp/llama-sycl-files.txt /tmp/ik-sycl-files.txt > /tmp/sycl-files-shared.txt
wc -l /tmp/sycl-files-only-in-llama.txt /tmp/sycl-files-shared.txt
```

Expected:

```text
The first count is greater than 0 and names /tmp/sycl-files-only-in-llama.txt.
The second count is greater than 0 and names /tmp/sycl-files-shared.txt.
```

- [ ] **Step 3: Record high-value missing files**

Run:

```bash
cd /root/git/ik_llama-arc-sycl-port
{
  echo
  echo "## Missing Source Files From llama.cpp"
  echo
  echo '```text'
  grep -E '^(cpy|set_rows|fattn|pool|conv2d|conv3d|outprod|element_wise|type|quants|quantize|dpct/helper)' /tmp/sycl-files-only-in-llama.txt || true
  echo '```'
} >> docs/superpowers/reports/2026-06-19-sycl-delta-inventory.md
git add docs/superpowers/reports/2026-06-19-sycl-delta-inventory.md
git commit -m "docs: inventory SYCL source deltas"
```

Expected:

```text
[arc-sycl-port ...] docs: inventory SYCL source deltas
```

## Task 3: Add SYCL Build Option Parity

**Files:**
- Modify: `ggml/CMakeLists.txt`
- Modify: `ggml/src/CMakeLists.txt`
- Modify: `docs/superpowers/reports/2026-06-19-sycl-delta-inventory.md`

- [ ] **Step 1: Add missing cache options**

In `ggml/CMakeLists.txt`, find the existing SYCL option block:

```cmake
option(GGML_SYCL                            "ggml: use SYCL"                                  OFF)
option(GGML_SYCL_F16                        "ggml: use 16 bit floats for sycl calculations"   OFF)
set   (GGML_SYCL_TARGET "INTEL" CACHE STRING
                                            "ggml: sycl target device")
```

Replace it with:

```cmake
option(GGML_SYCL                            "ggml: use SYCL"                                  OFF)
option(GGML_SYCL_F16                        "ggml: use 16 bit floats for sycl calculations"   OFF)
option(GGML_SYCL_GRAPH                      "ggml: enable graphs in the SYCL backend"         ON)
option(GGML_SYCL_HOST_MEM_FALLBACK          "ggml: allow host memory fallback in SYCL reorder (requires kernel 6.8+)" ON)
option(GGML_SYCL_SUPPORT_LEVEL_ZERO_API     "ggml: use Level Zero API in SYCL backend"        ON)
option(GGML_SYCL_DNN                        "ggml: enable oneDNN in the SYCL backend"         ON)
set   (GGML_SYCL_TARGET "INTEL" CACHE STRING
                                            "ggml: sycl target device")
set   (GGML_SYCL_DEVICE_ARCH "" CACHE STRING
                                            "ggml: sycl device architecture")
```

- [ ] **Step 2: Thread the new options into the SYCL CMake branch**

In `ggml/src/CMakeLists.txt`, inside `if (GGML_SYCL)`, after `add_compile_definitions(GGML_SYCL_WARP_SIZE=16)`, add:

```cmake
    if (GGML_SYCL_GRAPH)
        add_compile_definitions(GGML_SYCL_GRAPH)
    endif()

    if (GGML_SYCL_HOST_MEM_FALLBACK)
        add_compile_definitions(GGML_SYCL_HOST_MEM_FALLBACK)
    endif()

    if (GGML_SYCL_SUPPORT_LEVEL_ZERO_API)
        add_compile_definitions(GGML_SYCL_SUPPORT_LEVEL_ZERO_API)
    endif()

    if (GGML_SYCL_DNN)
        add_compile_definitions(GGML_SYCL_DNN)
    endif()

    if (GGML_SYCL_DEVICE_ARCH)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=spir64_gen -Xs \"-device ${GGML_SYCL_DEVICE_ARCH}\"")
    elseif (GGML_SYCL_TARGET STREQUAL "INTEL")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Xs -ze-intel-greater-than-4GB-buffer-required")
    endif()
```

- [ ] **Step 3: Verify configure option visibility**

Run:

```bash
cd /root/git/ik_llama-arc-sycl-port
git diff -- ggml/CMakeLists.txt ggml/src/CMakeLists.txt
grep -n "GGML_SYCL_HOST_MEM_FALLBACK\|GGML_SYCL_SUPPORT_LEVEL_ZERO_API\|GGML_SYCL_DEVICE_ARCH\|ze-intel-greater-than-4GB" ggml/CMakeLists.txt ggml/src/CMakeLists.txt
```

Expected:

```text
Output includes option(GGML_SYCL_HOST_MEM_FALLBACK in ggml/CMakeLists.txt.
Output includes option(GGML_SYCL_SUPPORT_LEVEL_ZERO_API in ggml/CMakeLists.txt.
Output includes set   (GGML_SYCL_DEVICE_ARCH in ggml/CMakeLists.txt.
Output includes if (GGML_SYCL_HOST_MEM_FALLBACK) in ggml/src/CMakeLists.txt.
Output includes if (GGML_SYCL_SUPPORT_LEVEL_ZERO_API) in ggml/src/CMakeLists.txt.
Output includes ze-intel-greater-than-4GB-buffer-required in ggml/src/CMakeLists.txt.
```

- [ ] **Step 4: Commit build option parity**

Run:

```bash
cd /root/git/ik_llama-arc-sycl-port
{
  echo
  echo "| Build options | llama.cpp ggml/CMakeLists.txt | ik_llama.cpp ggml/CMakeLists.txt | current commit | Enable Arc/B580 runtime switches and Level Zero controls. |"
} >> docs/superpowers/reports/2026-06-19-sycl-delta-inventory.md
git add ggml/CMakeLists.txt ggml/src/CMakeLists.txt docs/superpowers/reports/2026-06-19-sycl-delta-inventory.md
git commit -m "build: add SYCL Arc option parity"
```

Expected:

```text
[arc-sycl-port ...] build: add SYCL Arc option parity
```

## Task 4: Reapply Tensor Backend Correctness If Missing

**Files:**
- Modify: `ggml/src/ggml-sycl.cpp`
- Modify: `docs/superpowers/reports/2026-06-19-sycl-delta-inventory.md`

- [ ] **Step 1: Check current backend markings**

Run:

```bash
cd /root/git/ik_llama-arc-sycl-port
grep -n "ggml_backend_sycl_buffer_init_tensor\|GGML_BACKEND_TYPE_GPU\|GGML_BACKEND_TYPE_GPU_SPLIT" ggml/src/ggml-sycl.cpp
```

Expected:

```text
Output includes ggml_backend_sycl_buffer_init_tensor.
Output includes GGML_BACKEND_TYPE_GPU_SPLIT.
```

- [ ] **Step 2: Add normal SYCL tensor GPU marking when absent**

In `ggml/src/ggml-sycl.cpp`, inside `ggml_backend_sycl_buffer_init_tensor`, after the view-source early return block and before quantized padding initialization, ensure this exact line exists:

```cpp
    tensor->__backend = GGML_BACKEND_TYPE_GPU;
```

If that line already exists in the function, make no source change.

- [ ] **Step 3: Preserve split tensor marking**

In `ggml/src/ggml-sycl.cpp`, inside `ggml_backend_sycl_split_buffer_init_tensor`, ensure this exact line exists before assigning `tensor->extra`:

```cpp
    tensor->__backend = GGML_BACKEND_TYPE_GPU_SPLIT;
```

If that line already exists in the function, make no source change.

- [ ] **Step 4: Commit only if source changed**

Run:

```bash
cd /root/git/ik_llama-arc-sycl-port
git diff -- ggml/src/ggml-sycl.cpp
```

If the diff is non-empty, run:

```bash
{
  echo
  echo "| Tensor backend marking | old local fix-sycl-buffer-interface commits | ggml/src/ggml-sycl.cpp | current commit | Required so SYCL buffers are treated as GPU-backed tensors. |"
} >> docs/superpowers/reports/2026-06-19-sycl-delta-inventory.md
git add ggml/src/ggml-sycl.cpp docs/superpowers/reports/2026-06-19-sycl-delta-inventory.md
git commit -m "fix: mark SYCL tensors as GPU-backed"
```

If the diff is empty, run:

```bash
{
  echo
  echo "| Tensor backend marking | old local fix-sycl-buffer-interface commits | ggml/src/ggml-sycl.cpp | none | Already present on current origin/main. |"
} >> docs/superpowers/reports/2026-06-19-sycl-delta-inventory.md
git add docs/superpowers/reports/2026-06-19-sycl-delta-inventory.md
git commit -m "docs: record existing SYCL tensor marking"
```

Expected:

```text
[arc-sycl-port ...] fix: mark SYCL tensors as GPU-backed
```

or:

```text
[arc-sycl-port ...] docs: record existing SYCL tensor marking
```

## Task 5: Port Level Zero Allocation Hooks

**Files:**
- Modify: `ggml/src/CMakeLists.txt`
- Modify: `ggml/src/ggml-sycl.cpp`
- Modify: `ggml/src/ggml-sycl/common.cpp`
- Modify: `ggml/src/ggml-sycl/common.hpp`
- Modify: `docs/superpowers/reports/2026-06-19-sycl-delta-inventory.md`

- [ ] **Step 1: Locate Level Zero implementation in llama.cpp**

Run:

```bash
cd /root/git
grep -R -n "GGML_SYCL_SUPPORT_LEVEL_ZERO_API\|zeMemAllocDevice\|zeMemFree\|ze-intel-greater-than-4GB" llama.cpp/ggml/src/ggml-sycl llama.cpp/ggml/src/CMakeLists.txt llama.cpp/ggml/CMakeLists.txt > /tmp/llama-level-zero.txt
sed -n '1,220p' /tmp/llama-level-zero.txt
```

Expected:

```text
Output includes llama.cpp/ggml/src/ggml-sycl/CMakeLists.txt and GGML_SYCL_SUPPORT_LEVEL_ZERO_API.
```

- [ ] **Step 2: Locate current ik_llama.cpp implementation**

Run:

```bash
cd /root/git
grep -R -n "GGML_SYCL_SUPPORT_LEVEL_ZERO_API\|zeMemAllocDevice\|zeMemFree\|ze-intel-greater-than-4GB" ik_llama-arc-sycl-port/ggml/src ik_llama-arc-sycl-port/ggml/CMakeLists.txt > /tmp/ik-level-zero.txt || true
sed -n '1,220p' /tmp/ik-level-zero.txt
```

Expected before port:

```text
<empty or fewer matches than /tmp/llama-level-zero.txt>
```

- [ ] **Step 3: Port only the allocator dependency chain**

Copy the minimal Level Zero helper functions and compile guards from latest `llama.cpp` into the matching `ik_llama.cpp` SYCL files. Keep function names unchanged when the same names already exist in `ik_llama.cpp`. When a `llama.cpp` helper depends on unrelated operations, copy only the helper's declarations, definitions, and include lines required by the allocator path.

The port is complete when this command shows matches in `ik_llama.cpp`:

```bash
cd /root/git/ik_llama-arc-sycl-port
grep -R -n "GGML_SYCL_SUPPORT_LEVEL_ZERO_API\|zeMemAllocDevice\|zeMemFree" ggml/src/ggml-sycl.cpp ggml/src/ggml-sycl/common.cpp ggml/src/ggml-sycl/common.hpp ggml/src/CMakeLists.txt
```

Expected:

```text
Output includes GGML_SYCL_SUPPORT_LEVEL_ZERO_API in ggml/src/CMakeLists.txt.
Output includes zeMemAllocDevice in a ggml/src/ggml-sycl source file.
Output includes zeMemFree in a ggml/src/ggml-sycl source file.
```

- [ ] **Step 4: Commit Level Zero hook port**

Run:

```bash
cd /root/git/ik_llama-arc-sycl-port
{
  echo
  echo "| Level Zero allocation | llama.cpp ggml/src/ggml-sycl | ik_llama.cpp ggml/src/ggml-sycl | current commit | Avoids large Arc allocations going through slower or fragile generic SYCL paths. |"
} >> docs/superpowers/reports/2026-06-19-sycl-delta-inventory.md
git add ggml/src/CMakeLists.txt ggml/src/ggml-sycl.cpp ggml/src/ggml-sycl/common.cpp ggml/src/ggml-sycl/common.hpp docs/superpowers/reports/2026-06-19-sycl-delta-inventory.md
git commit -m "sycl: add Level Zero allocation support"
```

Expected:

```text
[arc-sycl-port ...] sycl: add Level Zero allocation support
```

## Task 6: Port Host Memory Fallback And Large Allocation Behavior

**Files:**
- Modify: `ggml/src/ggml-sycl.cpp`
- Modify: `ggml/src/ggml-sycl/common.cpp`
- Modify: `ggml/src/ggml-sycl/common.hpp`
- Modify: `docs/superpowers/reports/2026-06-19-sycl-delta-inventory.md`

- [ ] **Step 1: Find upstream fallback code**

Run:

```bash
cd /root/git
grep -R -n "GGML_SYCL_HOST_MEM_FALLBACK\|host memory fallback\|malloc_host\|malloc_shared\|system allocation" llama.cpp/ggml/src/ggml-sycl > /tmp/llama-host-fallback.txt
sed -n '1,220p' /tmp/llama-host-fallback.txt
```

Expected:

```text
Output includes GGML_SYCL_HOST_MEM_FALLBACK in a llama.cpp SYCL source file.
```

- [ ] **Step 2: Port fallback decision points**

Apply the fallback checks to the corresponding allocation/reorder paths in `ik_llama.cpp`. The port is complete when the target tree contains guarded uses of `GGML_SYCL_HOST_MEM_FALLBACK`:

```bash
cd /root/git/ik_llama-arc-sycl-port
grep -R -n "GGML_SYCL_HOST_MEM_FALLBACK" ggml/src/ggml-sycl.cpp ggml/src/ggml-sycl/common.cpp ggml/src/ggml-sycl/common.hpp
```

Expected:

```text
Output includes GGML_SYCL_HOST_MEM_FALLBACK in an ik_llama.cpp SYCL source file.
```

- [ ] **Step 3: Commit host memory fallback port**

Run:

```bash
cd /root/git/ik_llama-arc-sycl-port
{
  echo
  echo "| Host memory fallback | llama.cpp ggml/src/ggml-sycl | ik_llama.cpp ggml/src/ggml-sycl | current commit | Allows inference to continue when Arc device memory is exhausted during reorder. |"
} >> docs/superpowers/reports/2026-06-19-sycl-delta-inventory.md
git add ggml/src/ggml-sycl.cpp ggml/src/ggml-sycl/common.cpp ggml/src/ggml-sycl/common.hpp docs/superpowers/reports/2026-06-19-sycl-delta-inventory.md
git commit -m "sycl: add host memory fallback path"
```

Expected:

```text
[arc-sycl-port ...] sycl: add host memory fallback path
```

## Task 7: Decide On Performance Kernel Ports

**Files:**
- Modify: `docs/superpowers/reports/2026-06-19-sycl-delta-inventory.md`
- Optional Modify: `ggml/src/ggml-sycl/*.cpp`
- Optional Modify: `ggml/src/ggml-sycl/*.hpp`

- [ ] **Step 1: Identify Arc-relevant kernel changes**

Run:

```bash
cd /root/git
git -C llama.cpp log --oneline --all --grep='SYCL\\|sycl\\|Arc\\|B580\\|Level Zero\\|Q4\\|Q5\\|Q6\\|MoE' -- ggml/src/ggml-sycl > /tmp/llama-sycl-kernel-log.txt
sed -n '1,80p' /tmp/llama-sycl-kernel-log.txt
```

Expected:

```text
Output includes one or more SYCL commits touching ggml/src/ggml-sycl.
```

- [ ] **Step 2: Classify each kernel area**

Run this command to append a concrete decision table to `docs/superpowers/reports/2026-06-19-sycl-delta-inventory.md`:

```bash
cd /root/git/ik_llama-arc-sycl-port
q_commit=$(grep -Ei 'Q4|Q5|Q6|mul_mat|MMQ|MMVQ' /tmp/llama-sycl-kernel-log.txt | head -1 | cut -d' ' -f1)
moe_commit=$(grep -Ei 'MoE|prefill|MUL_MAT_ID' /tmp/llama-sycl-kernel-log.txt | head -1 | cut -d' ' -f1)
vision_commit=$(grep -Ei 'conv|pool' /tmp/llama-sycl-kernel-log.txt | head -1 | cut -d' ' -f1)
fattn_commit=$(grep -Ei 'fattn|flash' /tmp/llama-sycl-kernel-log.txt | head -1 | cut -d' ' -f1)
{
  echo
  echo "## Performance Kernel Candidates"
  echo
  echo "| Area | llama.cpp commit | Port now | Reason |"
  echo "| --- | --- | --- | --- |"
  echo "| Q4/Q5/Q6 quantized matmul | ${q_commit:-not-found} | yes | Common quantized LLM path on Arc. |"
  echo "| MoE prefill | ${moe_commit:-not-found} | yes | Important for MoE models when loaded through ik_llama.cpp. |"
  echo "| Conv/pool vision ops | ${vision_commit:-not-found} | no | Not required for initial text server validation. |"
  echo "| Flash attention template expansion | ${fattn_commit:-not-found} | no | Baseline server run should be stable before flash-attention template expansion. |"
} >> docs/superpowers/reports/2026-06-19-sycl-delta-inventory.md
```

- [ ] **Step 3: Commit the performance-kernel decision before code ports**

Run:

```bash
cd /root/git/ik_llama-arc-sycl-port
git add docs/superpowers/reports/2026-06-19-sycl-delta-inventory.md
git commit -m "docs: classify SYCL kernel port candidates"
```

Expected:

```text
[arc-sycl-port ...] docs: classify SYCL kernel port candidates
```

## Task 8: Update SYCL Documentation For B580 Host

**Files:**
- Modify: `docs/backend/SYCL.md`
- Create: `docs/superpowers/reports/2026-06-19-b580-validation.md`

- [ ] **Step 1: Update supported GPU table**

In `docs/backend/SYCL.md`, find the Intel Arc support table and ensure it includes:

```markdown
| Intel Arc B-Series            | Support | Arc B580                              |
```

If the table already includes Arc B580, leave it unchanged.

- [ ] **Step 2: Create B580 validation checklist**

Create `docs/superpowers/reports/2026-06-19-b580-validation.md` with:

````markdown
# Arc B580 Validation

## Host

- CPU: Intel i9, 20 hardware threads
- GPU: Intel Arc B580
- RAM: 64 GB
- Backend: SYCL / Level Zero

## Device Check

```bash
source /opt/intel/oneapi/setvars.sh
sycl-ls
```

Expected evidence: output includes a Level Zero GPU entry for Intel Arc B580.

## Build

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build-sycl-b580 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_SYCL=ON \
  -DGGML_SYCL_F16=ON \
  -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON \
  -DGGML_SYCL_HOST_MEM_FALLBACK=ON \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx
cmake --build build-sycl-b580 --target llama-server llama-cli llama-bench -j20
```

Expected evidence: build completes and produces `build-sycl-b580/bin/llama-server`.

## Device Listing

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build-sycl-b580/bin/llama-server --list-devices
```

Expected evidence: output lists a SYCL device for Intel Arc B580.

## Benchmark

| Binary | Model Path | Prompt Tokens/s | Generation Tokens/s | Command |
| --- | --- | ---: | ---: | --- |
| ik_llama.cpp SYCL FP16 | `/my_local_files/gguf/Qwen_Qwen3-0.6B-IQ4_NL.gguf` | 0 | 0 | `./build-sycl-b580/bin/llama-bench -m /my_local_files/gguf/Qwen_Qwen3-0.6B-IQ4_NL.gguf -ngl 999` |
| ik_llama.cpp SYCL FP32 | `/my_local_files/gguf/Qwen_Qwen3-0.6B-IQ4_NL.gguf` | 0 | 0 | `./build-sycl-b580-fp32/bin/llama-bench -m /my_local_files/gguf/Qwen_Qwen3-0.6B-IQ4_NL.gguf -ngl 999` |
| llama.cpp SYCL FP16 | `/my_local_files/gguf/Qwen_Qwen3-0.6B-IQ4_NL.gguf` | 0 | 0 | `./build-sycl-b580/bin/llama-bench -m /my_local_files/gguf/Qwen_Qwen3-0.6B-IQ4_NL.gguf -ngl 999` |
````

- [ ] **Step 3: Commit docs**

Run:

```bash
cd /root/git/ik_llama-arc-sycl-port
git add docs/backend/SYCL.md docs/superpowers/reports/2026-06-19-b580-validation.md
git commit -m "docs: add Arc B580 SYCL validation"
```

Expected:

```text
[arc-sycl-port ...] docs: add Arc B580 SYCL validation
```

## Task 9: Coding Container Verification

**Files:**
- No source files modified unless formatting tools change files.

- [ ] **Step 1: Run source searches for required knobs**

Run:

```bash
cd /root/git/ik_llama-arc-sycl-port
grep -R -n "GGML_SYCL_SUPPORT_LEVEL_ZERO_API\|GGML_SYCL_HOST_MEM_FALLBACK\|GGML_SYCL_DEVICE_ARCH" ggml docs | head -80
grep -R -n "GGML_BACKEND_TYPE_GPU" ggml/src/ggml-sycl.cpp | head -20
```

Expected:

```text
Output includes option(GGML_SYCL_HOST_MEM_FALLBACK in ggml/CMakeLists.txt.
Output includes option(GGML_SYCL_SUPPORT_LEVEL_ZERO_API in ggml/CMakeLists.txt.
Output includes set   (GGML_SYCL_DEVICE_ARCH in ggml/CMakeLists.txt.
```

- [ ] **Step 2: Run non-SYCL configure smoke if CMake is available**

Run:

```bash
cd /root/git/ik_llama-arc-sycl-port
if command -v cmake >/dev/null 2>&1; then
  cmake -S . -B build-cpu-smoke -DCMAKE_BUILD_TYPE=Release
else
  echo "cmake unavailable in coding container; skipping configure smoke"
fi
```

Expected:

```text
-- Configuring done
```

or:

```text
cmake unavailable in coding container; skipping configure smoke
```

- [ ] **Step 3: Run SYCL configure smoke if oneAPI is available**

Run:

```bash
cd /root/git/ik_llama-arc-sycl-port
if [ -f /opt/intel/oneapi/setvars.sh ] && command -v cmake >/dev/null 2>&1; then
  . /opt/intel/oneapi/setvars.sh
  cmake -S . -B build-sycl-smoke -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_SYCL=ON \
    -DGGML_SYCL_F16=ON \
    -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON \
    -DGGML_SYCL_HOST_MEM_FALLBACK=ON \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx
else
  echo "oneAPI or cmake unavailable in coding container; skipping SYCL configure smoke"
fi
```

Expected:

```text
-- Configuring done
```

or:

```text
oneAPI or cmake unavailable in coding container; skipping SYCL configure smoke
```

- [ ] **Step 4: Final status**

Run:

```bash
cd /root/git/ik_llama-arc-sycl-port
git status --short --branch
git log --oneline origin/main..HEAD
```

Expected:

```text
## arc-sycl-port
<commits from this plan>
```
