# Developer guide — policy execution pipeline

For maintainers, contributors, and downstream users integrating these ROS 2 packages.

This guide covers how to use the APIs and where responsibilities sit. Package names, short descriptions, and build commands live in the repository [README](../README.md). 

## Using the core

- Depend on `ros2_policy_execution_core` in `package.xml`; wire CMake with `ament_target_dependencies` / `find_package` as usual.
- Flow: preprocessor → `InferenceCore::run_inference(NamedValueList in, NamedValueList out)` → `PostprocessorCore::process(NamedValueList)`. After `build_observation()`, use `get_observation_named_value_list()` (includes tensor `observation`).

### Recommended header includes

Tensor and named-value types live under `include/ros2_policy_execution_core/tensor/`. Include them as `#include "ros2_policy_execution_core/tensor/<header>.hpp"`.

| Header | Use for |
|--------|---------|
| `tensor/data_type.hpp` | `DataType`, `DataTypeForElement`, `data_type_v`, `data_type_size` only |
| `tensor/tensor_device.hpp` | `Device` / `DeviceType` only |
| `tensor/tensor_types.hpp` | `ByteBufferView`, `Tensor` (pulls in `data_type.hpp`) |
| `tensor/named_value_list.hpp` | `Value`, `NamedValueList`, `find_value` |
| `tensor/value_types.hpp` | Umbrella (all of the above) |

Each observation provider returns `ObservationData`: a timestamp and a `Value` whose active payload is a contiguous `Tensor` with element count matching that provider’s registered segment names. Today `PreprocessorCore::build_observation` only accepts `DataType::Float32` tensors because it concatenates into a `std::vector<float>` before wrapping the combined observation for inference. Use `observation_data_from_floats()` or `observation_data_from_float_vector()` when the source is `std::vector<float>`; otherwise construct `Value` / `Tensor` like the rest of the pipeline. See `preprocessor_support.hpp` and `ros2_policy_execution_core/README.md`.

## Core vs adapters (boundary)

| Layer | Owns | Must not leak into core’s public API |
|-------|------|--------------------------------------|
| Core | Stage abstractions (`PreprocessorCore`, `InferenceCore`, `PostprocessorCore`), `Tensor` / `ByteBufferView` / `NamedValueList`, `ObservationData`, history helpers. | — |
| Adapters | Conversion to/from vendor tensors (`Ort::Value`, TRT bindings, `ov::Tensor`, etc.), runtime session/options, versioned SDK linkage. | Vendor types and headers stay in adapter packages and downstream nodes that opt in. |
| Your stack | Concrete preprocessors, inference engines, postprocessors (often subclasses in separate repos or packages). | — |

The pipeline contract between stages is `Tensor` and `NamedValueList` only. Adapters sit at the edges: core → adapter → runtime, and back.

## Runtimes

Core ships no ONNX/TRT/OpenVINO/TFLite headers or libraries. Optional adapter packages depend on `ros2_policy_execution_core` and should include `tensor/tensor_types.hpp` (or `tensor/named_value_list.hpp`) only where they touch that contract; dtype-only mapping (e.g. to a runtime enum) can use `tensor/data_type.hpp` alone. Additional backends: new sibling packages, same boundary as `ros2_policy_execution_adapters`.

## Design principles

1. Stable transport contract — Treat `DataType`, `ByteBufferView`, `Tensor`, and `NamedValueList` as the long-lived surface; extend compatibly (e.g. new `Value` kinds) rather than breaking callers silently.
2. Adapters at the edge — No vendor SDKs inside core; keep conversion and allocator details in adapter or application packages.
3. Tests match the split — Core tests: buffers, shapes, dtypes, `NamedValueList` behavior, preprocessor rules. Adapter / integration tests: next to the runtime they need.
4. Defer until needed — e.g. a dedicated strided tensor view (core `Tensor` is dense row-major only), automatic NCHW/NHWC fixes, GPU allocation inside core types.

## Deferred: strided tensor support

The core `Tensor` type enforces dense, contiguous, row-major storage. Strides — per-dimension byte steps that can differ from the dense layout — are intentionally not supported yet. In frameworks like NumPy and PyTorch, strides enable zero-copy transpose, slicing, and broadcasting, but they add complexity to every consumer of the tensor (each pipeline stage must handle non-contiguous memory or call `.contiguous()` before proceeding).

The current pipeline contract passes contiguous buffers between preprocessor, inference, and postprocessor stages, and the ORT adapter requires contiguous data for zero-copy wrapping. Strides would only provide a benefit if profiling showed that inter-stage copies are a bottleneck worth eliminating by passing non-contiguous views through the pipeline. Until that need is demonstrated, the simpler dense-only invariant keeps the API surface small and every stage's assumptions uniform. If strided support is added later, it should be introduced as a separate type (e.g. `StridedTensorView`) rather than complicating the existing `Tensor` class.

## Versioning approach for core types

Core types (`Tensor`, `ByteBufferView`, `NamedValueList`, `DataType`) carry no version field. The approach is:

- Evolve in-place for additive changes — new `DataType` values, new `Device` capabilities, new fields with defaults. These don't break existing callers.
- New types for different contracts — fundamentally different representations (e.g. `StridedTensorView`, `SparseTensor`) should be introduced as new `Value::Kind` variants in the existing `std::variant`, not as version bumps on `Tensor`.
- Version the package, not the struct — use semver on the package for breaking changes. In-process types that we control don't benefit from per-struct version fields, which add dispatch complexity (`if version >= 2`) with no real gain.
- Version at the serialization boundary — if tensors ever cross process boundaries (shared memory, ROS topics), version the wire format at that layer, not in the core in-memory types.

## Maintainers

Prefer the principles above when reviewing changes. Paths: `ros2_policy_execution_core/include/ros2_policy_execution_core/tensor/`, `ros2_policy_execution_adapters/include/.../ort_tensor_conversion.hpp`.
