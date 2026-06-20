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

## Missing Source Files From llama.cpp

```text
conv2d-dw.cpp
conv2d-dw.hpp
conv2d-transpose.cpp
conv2d-transpose.hpp
conv2d.cpp
conv2d.hpp
conv3d.cpp
conv3d.hpp
cpy.cpp
cpy.hpp
element_wise.cpp
element_wise.hpp
fattn-buffers.cpp
fattn-buffers.hpp
fattn-common.hpp
fattn-tile.cpp
fattn-tile.hpp
fattn-vec.hpp
fattn.cpp
fattn.hpp
outprod.cpp
outprod.hpp
pool.cpp
pool.hpp
quantize.hpp
quants.hpp
set_rows.cpp
set_rows.hpp
type.hpp
```

| Build options | llama.cpp ggml/CMakeLists.txt | ik_llama.cpp ggml/CMakeLists.txt | current commit | Enable Arc/B580 runtime switches and Level Zero controls. |

| Tensor backend marking | old local fix-sycl-buffer-interface commits | ggml/src/ggml-sycl.cpp | current commit | Required so SYCL buffers are treated as GPU-backed tensors. |

| Level Zero allocation | llama.cpp ggml/src/ggml-sycl | ik_llama.cpp ggml/src/ggml-sycl | current commit | Avoids large Arc allocations going through slower or fragile generic SYCL paths. |

| Host memory fallback | llama.cpp ggml/src/ggml-sycl | ik_llama.cpp ggml/src/ggml-sycl | current commit | Allows inference to continue when Arc device memory is exhausted during normal buffer allocation. |

## Performance Kernel Candidates

| Area | llama.cpp commit | Port now | Reason |
| --- | --- | --- | --- |
| Q4/Q5/Q6 quantized matmul | dd69db292 | yes | Common quantized LLM path on Arc. |
| MoE prefill | ebbc1e51c | yes | Important for MoE models when loaded through ik_llama.cpp. |
| Conv/pool vision ops | 6f1034b32 | no | Not required for initial text server validation. |
| Flash attention template expansion | a51142497 | no | Baseline server run should be stable before flash-attention template expansion. |
