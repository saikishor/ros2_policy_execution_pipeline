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

### ObservationData

Providers return `ObservationData`: a timestamp and a `Value` holding a contiguous `DataType::Float32` tensor
with one element per registered segment name. Helpers avoid duplicating the vector-to-tensor wiring:

```cpp
// Owns floats in a new shared vector (copies or moves from a temporary vector).
ObservationData d1 = observation_data_from_floats({1.0f, 2.0f}, stamp);

// Reuses existing shared vector storage (no copy of elements).
auto storage = std::make_shared<std::vector<float>>(...);
ObservationData d2 = observation_data_from_float_vector(storage, stamp);

// Same transport type as inference I/O: build Value / Tensor directly when already in tensor form.
Value v(Tensor::share_vector(storage, {static_cast<int64_t>(storage->size())}));
ObservationData d3(v, stamp);
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

Headers are layered so you can depend only on what you need (see [developer guide](../doc/developer_guide.md) at repository root):

- `tensor/data_type.hpp` — `DataType`, `DataTypeForElement`, `data_type_v`, `data_type_size`
- `tensor/tensor_device.hpp` — `Device` / `DeviceType`
- `tensor/tensor_types.hpp` — `ByteBufferView`, `Tensor` (includes `data_type.hpp`)
- `tensor/named_value_list.hpp` — `Value`, `NamedValue`, `NamedValueList`, `find_value`
- `tensor/value_types.hpp` — umbrella that includes all of the above

Each path is included as `ros2_policy_execution_core/tensor/...`.

The important ownership idea is unchanged:

- `ByteBufferView`: non-owning `(data, bytes)` plus optional `shared_ptr` anchor for lifetime
- `Tensor`: dense tensor metadata plus a `ByteBufferView` into the payload bytes
- `Value`: extensible wrapper around a `Tensor`
- `NamedValue` / `NamedValueList`: ordered named inputs and outputs

That means a preprocessor can often wrap existing memory directly, for example:

- a `std::vector<float>`
- a ROS image message payload
- a compact array built from pose data

without copying the payload again.

Inference adapters (ONNX Runtime, TensorRT, OpenVINO, TFLite, etc.) are intentionally **not**
part of this package; add them in a separate package that depends on these headers. See
[`doc/developer_guide.md`](../doc/developer_guide.md) at repository root.

#### `ByteBufferView` construction example

The most common pattern is to keep the original `std::shared_ptr` in the caller and let the
buffer store another shared handle to the same payload:

```cpp
auto values = std::make_shared<std::vector<float>>(
  std::initializer_list<float>{1.0f, 2.0f, 3.0f});

ByteBufferView buffer(values->data(), values->size() * sizeof(float), values);

// `values` is still valid here.
// `buffer.owner()` also keeps the same vector storage alive.
```

Only the smart-pointer handle is shared or moved, never the payload bytes themselves.
If the caller uses `std::move(values)`, then the caller gives its handle to the buffer and
`values` becomes empty.

## API Reference

### InferenceCore

| Method | Description |
|--------|-------------|
| `run_inference(inputs, outputs)` | Run inference on a `NamedValueList` of named inputs; fill `outputs` |

### PostprocessorCore

| Method | Description |
|--------|-------------|
| `process(inference_output)` | Postprocess inference `NamedValueList`; return command `NamedValueList` |

### PreprocessorCore

| Method | Description |
|--------|-------------|
| `register_observation_provider(name, provider)` | Register a named observation data provider |
| `build_observation(current_time)` | Build observation by calling all providers |
| `get_observation()` | Get the built observation vector |
| `get_observation_named_value_list()` | Get named tensors for inference (`observation` float vector) |
| `get_observation_time_diffs()` | Get map of provider names to data age (seconds) |
| `has_observation_providers()` | Check if any providers are registered |
| `clear_observation_providers()` | Remove all registered providers |
| `set_config(config)` | Set history length configuration |
| `set_previous_observations(obs)` | Add observation to history |
| `set_previous_actions(action)` | Add action to history |
| `get_observation_history()` | Get observation history (newest first) |
| `get_action_history()` | Get action history (newest first) |

