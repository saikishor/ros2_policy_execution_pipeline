# ros2_policy_execution_core

Core library for the ROS 2 policy execution pipeline. Provides base classes for building a complete neural network policy inference pipeline with three stages:

1. **PreprocessorCore** - Collect and prepare observation data
2. **InferenceCore** - Run neural network inference
3. **PostprocessorCore** - Transform actions into final commands

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Preprocessor   │────▶│    Inference    │────▶│  Postprocessor   │
│                 │     │                 │     │                  │
│ - Collect obs   │     │ - Run NN model  │     │ - Scale actions  │
│ - Track time    │     │ - Produce raw   │     │ - Apply limits   │
│ - Build vector  │     │   actions       │     │ - Final commands │
└─────────────────┘     └─────────────────┘     └──────────────────┘
```

---

## InferenceCore

Abstract base class for neural network inference engines. Implement this to integrate different ML frameworks (ONNX, TensorRT, PyTorch, etc.).


---

## PostprocessorCore

Abstract base class for action postprocessing. Implement this to apply scaling, limiting, smoothing, or other transformations to raw policy outputs.


---

## PreprocessorCore

### Observation Providers

Register callbacks that provide observation data. Providers are called in registration order, and their outputs are concatenated to form the final observation vector.

### ObservationData Structure

Providers return `ObservationData` containing:
- `values`: const reference to the observation vector
- `timestamp`: `rclcpp::Time` when the data was captured

```cpp
struct ObservationData
{
  const std::vector<double>& values;
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

### Value Types

The package now also provides a backend-agnostic transport type for future preprocessors,
inference backends, and postprocessors.

The core types live in `value_types.hpp`:

- `SharedBuffer`: pointer + byte count + optional shared owner
- `Tensor`: dense tensor metadata plus a view into a `SharedBuffer`
- `Value`: extensible wrapper around a `Tensor`
- `NamedValue` / `ValueSet`: ordered named inputs and outputs

The important ownership idea is:

- the tensor does not need to own the payload bytes itself
- it may instead borrow a pointer to bytes owned by something else
- the optional `owner` field keeps that external object alive

That means a preprocessor can often wrap existing memory directly, for example:

- a `std::vector<float>`
- a ROS image message payload
- a compact array built from pose data

without copying the payload again.

ONNX Runtime interop lives separately in `ort_value_conversion.hpp`.
For dense CPU tensors, the conversion layer can wrap the same payload bytes as an `Ort::Value`
when ONNX Runtime allows caller-owned buffers.

#### `SharedBuffer` Construction Example

The most common pattern is to keep the original `std::shared_ptr` in the caller and let the
buffer store another shared handle to the same payload:

```cpp
auto values = std::make_shared<std::vector<float>>(
  std::initializer_list<float>{1.0f, 2.0f, 3.0f});

SharedBuffer buffer(values->data(), values->size() * sizeof(float), values);

// `values` is still valid here.
// `buffer.owner()` also keeps the same vector storage alive.
```

Only the smart-pointer handle is shared or moved, never the payload bytes themselves.
If the caller uses `std::move(values)`, then the caller gives its handle to the buffer and
`values` becomes empty.

#### Examples

Two small examples are provided in `examples/`:

- `value_types_basic_example.cpp`
  Shows how to wrap a shared vector as a `Tensor`, convert it to `Ort::Value`,
  and convert it back without copying the payload.
- `ros2_preprocessor_value_example.cpp`
  Shows a ROS 2 node that subscribes to image and pose topics and condenses them into a
  backend-agnostic `ValueSet`.

Run the generic datatype + ONNX example with:

```bash
ros2 run ros2_policy_execution_core value_types_basic_example
```

Run the ROS 2 preprocessor-style example with:

```bash
ros2 run ros2_policy_execution_core ros2_preprocessor_value_example
```

To exercise the ROS 2 example, publish some input data in separate terminals after sourcing the
workspace:

```bash
ros2 topic pub --once /tcp_pose geometry_msgs/msg/PoseStamped \
  "{pose: {position: {x: 0.1, y: 0.2, z: 0.3}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}"
```

```bash
ros2 topic pub --once /image sensor_msgs/msg/Image \
  "{height: 1, width: 2, encoding: 'rgb8', step: 6, data: [255, 0, 0, 0, 255, 0]}"
```

The basic ONNX example is only built when ONNX Runtime is available.
In this environment it is expected under `/opt/onnxruntime/`.

## API Reference

### InferenceCore

| Method | Description |
|--------|-------------|
| `run_inference(obs, action)` | Run inference on observations, populate action vector |

### PostprocessorCore

| Method | Description |
|--------|-------------|
| `process(actions)` | Process raw actions and return final commands |

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

