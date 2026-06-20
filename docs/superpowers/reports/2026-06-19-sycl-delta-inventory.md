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
