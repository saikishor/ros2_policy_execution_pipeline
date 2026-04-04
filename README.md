# ros2_policy_execution_pipeline

ROS 2 packages for a policy execution pipeline (preprocess → infer → postprocess).

## Packages

Canonical list of workspace packages (names, roles, key entry points):

| Package | Role |
|---------|------|
| `ros2_policy_execution_core` | Base types, preprocessor / inference / postprocessor interfaces, `NamedValueList` transport |
| `ros2_policy_execution_adapters` | Optional ONNX Runtime ↔ `Tensor` conversion (`ort_tensor_conversion.hpp`); requires ONNX Runtime at configure time |

Build only the core package if ONNX Runtime is not installed:

```bash
colcon build --packages-select ros2_policy_execution_core
```

See [`doc/developer_guide.md`](doc/developer_guide.md) for layering and design notes.
