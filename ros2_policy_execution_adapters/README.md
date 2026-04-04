# ros2_policy_execution_adapters

Header-only helpers that connect **ros2_policy_execution_core** tensor types to optional third-party runtimes.

This package converts **`Tensor` ↔ `Ort::Value`** only. Pipeline types such as **`NamedValueList`** live in core; you map each named entry to ORT inputs/outputs in your inference node or plugin.

## ONNX Runtime

- Header: `ros2_policy_execution_adapters/ort_tensor_conversion.hpp`
- Namespace: `ros2_policy_execution_adapters`
- Depends on the ROS 2 package **onnxruntime_vendor** (ONNX Runtime C++ API, `onnxruntime_cxx_api.h`); CMake links `onnxruntime::onnxruntime`.

Add `<depend>onnxruntime_vendor</depend>` (or ensure it is in your workspace) so `colcon` builds it before this package.

Your target should depend on `ros2_policy_execution_adapters`; the imported target links ONNX Runtime transitively when using `ament_cmake` exports.
