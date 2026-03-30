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
