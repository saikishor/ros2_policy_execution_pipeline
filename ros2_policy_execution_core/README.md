# ros2_policy_execution_core

Core library for the ROS 2 policy execution pipeline. Provides base classes for building a complete neural network policy inference pipeline with four stages:

1. **PreprocessorCore** - Collect and prepare observation data
2. **InferenceCore** - Run neural network inference
3. **PostprocessorCore** - Transform actions into final commands
4. **ExecutorCore** - Send the final commands to the robot

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Preprocessor   │────▶│    Inference    │────▶│  Postprocessor   │────▶│     Executor     │
│                 │     │                 │     │                  │     │                  │
│ - Collect obs   │     │ - Run NN model  │     │ - Scale actions  │     │ - Publish        │
│ - Track time    │     │ - Produce raw   │     │ - Apply limits   │     │   commands       │
│ - Build vector  │     │   actions       │     │ - Final commands │     │ - Write command  │
│                 │     │                 │     │                  │     │   interfaces     │
└─────────────────┘     └─────────────────┘     └──────────────────┘     └──────────────────┘
```

---

## Common Types

`onnxruntime_types.hpp` provides a shared pointer alias used throughout the pipeline:

```cpp
#include "ros2_policy_execution_core/onnxruntime_types.hpp"

// ros2_policy_execution_core::OrtValueSharedPtr
using OrtValueSharedPtr = std::shared_ptr<Ort::Value>;
```

All four pipeline stages pass data as `std::vector<OrtValueSharedPtr>`, avoiding copies between stages.

### Why ONNX Runtime? (Datatype, not framework lock-in)

The dependency on ONNX Runtime is a dependency on its **datatype** (`Ort::Value`), not on ONNX as the inference framework. `Ort::Value` is used purely as the shared, backend-agnostic data contract between the preprocessor, inference, and postprocessor stages because it natively represents every tensor element type the pipeline needs (`float`, `uint8_t` for images, `int64_t`, `bool`, `float16`, …) with type-safe typed accessors and solves history-buffer ownership via `shared_ptr<Ort::Value>` + ORT's allocator.

This does **not** mean the pipeline is ONNX-only. Other inference frameworks — TensorRT, OpenVINO, PyTorch, JAX, etc. — are fully supported as long as their data is converted to and from `Ort::Value` inside the `InferenceCore` implementation:

```
Ort::Value → (framework-specific tensor) → Inference → (framework-specific tensor) → Ort::Value
```

The inference node is the only framework-specific component. Preprocessors and postprocessors are identical across all backends, since they only ever see `Ort::Value`. The trade-off is that ONNX Runtime is a compile-time dependency for all pipeline plugins (even framework-agnostic preprocessors/postprocessors and tests), because they share the `Ort::Value` contract type.

---

## InferenceCore

Abstract base class for neural network inference engines. Implement this to integrate different ML frameworks (ONNX, TensorRT, PyTorch, OpenVINO, JAX, etc.). This is the **only** framework-specific stage: a non-ONNX backend converts the incoming `Ort::Value` inputs to its own tensor type, runs inference, and converts the results back to `Ort::Value` (see [Why ONNX Runtime?](#why-onnx-runtime-datatype-not-framework-lock-in)).

Inputs and outputs are passed as `std::vector<OrtValueSharedPtr>` (the `Ort::Value` data contract), keeping data on the device without copies between pipeline stages.

---

## PostprocessorCore

Abstract base class for action postprocessing. Implement this to apply scaling, limiting, smoothing, or other transformations to raw policy outputs.

Input and output use `std::vector<OrtValueSharedPtr>` to stay consistent with the tensor types produced by `InferenceCore`.

---

## ExecutorCore

Abstract base class for the terminal stage of the pipeline. Implement this to send the final command tensors produced by `PostprocessorCore` to the robot, either by publishing them to ROS topics or by writing them to ros2_control command interfaces.

Input uses `std::vector<OrtValueSharedPtr>` to stay consistent with the tensor types produced by `PostprocessorCore`. The base class is deliberately free of any `ros2_control` dependency (no `hardware_interface`/`controller_interface`): a command-interface executor holds its own `hardware_interface::LoanedCommandInterface` references, obtained from whatever owns it (e.g. the controller instantiating the pipeline).

---

## PreprocessorCore

### Observation Providers

Register callbacks that provide observation data. Providers are called in registration order, and their outputs are concatenated to form the final observation vector.

### ObservationData Structure

Providers return `ObservationData` containing:
- `values`: const reference to the observation tensors
- `timestamp`: `rclcpp::Time` when the data was captured

```cpp
struct ObservationData
{
  const std::vector<OrtValueSharedPtr> & values;
  rclcpp::Time timestamp;
};
```

### History Management

Configure and use observation/action history for sequence-based policies:

```cpp
PreprocessorCoreConfig config;
config.observation_history_length = 10;  // Keep last 10 observations
config.action_history_length = 5;        // Keep last 5 actions
preprocessor.set_config(config);

// After each inference step
preprocessor.set_previous_observations(current_obs);
preprocessor.set_previous_actions(current_action);

// Access history (most recent first)
const auto& obs_history = preprocessor.get_observation_history();
const auto& action_history = preprocessor.get_action_history();
```

## API Reference

### InferenceCore

| Method | Description |
|--------|-------------|
| `run_inference(obs, output)` | Run inference on `OrtValueSharedPtr` observation tensors, populate output tensors |

### PostprocessorCore

| Method | Description |
|--------|-------------|
| `process(inference_output)` | Process `OrtValueSharedPtr` inference tensors and return final command tensors |

### ExecutorCore

| Method | Description |
|--------|-------------|
| `execute(commands)` | Send the final `OrtValueSharedPtr` command tensors to the robot; returns true on success |

### PreprocessorCore

| Method | Description |
|--------|-------------|
| `register_observation_provider(name, provider)` | Register a named observation data provider |
| `build_observation(current_time)` | Build observation by calling all providers |
| `get_observation()` | Get the built observation vector |
| `get_observation_time_diffs()` | Get map of provider names to data age (seconds) |
| `has_observation_providers()` | Check if any providers are registered |
| `clear_observation_providers()` | Remove all registered providers |
| `set_config(config)` | Set history length configuration |
| `set_previous_observations(obs)` | Add observation to history |
| `set_previous_actions(action)` | Add action to history |
| `get_observation_history()` | Get observation history (newest first) |
| `get_action_history()` | Get action history (newest first) |

---

## Running the pipeline

This package only provides the abstract base classes. To actually run a pipeline, implement each
stage as a `pluginlib` plugin and use the
[`ros2_policy_execution_pipeline`](../ros2_policy_execution_pipeline) package's
`policy_execution_pipeline_node`, which loads the four plugins by name (from parameters) and drives
the preprocess → infer → postprocess → execute cycle on a timer at a configurable rate. See that
package's README for the parameter reference and an example configuration.
