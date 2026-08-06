# ros2_policy_execution_pipeline

Runtime node for the [`ros2_policy_execution_core`](../ros2_policy_execution_core) pipeline. Loads
one `pluginlib` plugin per pipeline stage — preprocessor, inference, postprocessor, executor —
configures them, and drives the full cycle on a timer at a configurable frequency.

## What it does

`PolicyExecutionPipeline::init()`:

1. Reads the four plugin-name parameters and the update rate.
2. Loads each plugin via a `pluginlib::ClassLoader<...Core>`, using
   `ros2_policy_execution_core` as the base package.
3. Applies the observation/action history configuration to the preprocessor.
4. Calls `configure(node)` on all four plugins.
5. Starts a wall timer that calls `update()` at `1.0 / update_rate` seconds.

Each `update()` cycle, in order:

1. `PreprocessorCore::build_observation(now)` → `get_observation()`
2. `InferenceCore::run_inference(observation, inference_output)`
3. `PostprocessorCore::process(inference_output)` → commands
4. `ExecutorCore::execute(commands)`
5. On full success, feeds the history: `set_previous_observations(observation)` and
   `set_previous_actions(inference_output)` — the **raw inference output**, not the postprocessed
   commands, since that is the policy's own action in its own output space.

Any stage returning `false`, or any exception raised anywhere in the cycle (in particular
`PreprocessorCore::build_observation()`, which signals invalid observation data by throwing
`std::runtime_error` rather than returning `false`), aborts just that cycle: it is logged at a
throttled rate and the node keeps running. History is only updated for a cycle that completed
observation building and inference successfully, so a failed cycle never poisons the history
buffers with partial data.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `preprocessor_plugin` | string | *(required)* | Plugin lookup name, e.g. `my_pkg/MyPreprocessor` |
| `inference_plugin` | string | *(required)* | Plugin lookup name |
| `postprocessor_plugin` | string | *(required)* | Plugin lookup name |
| `executor_plugin` | string | *(required)* | Plugin lookup name |
| `update_rate` | double | `100.0` | Pipeline update rate in Hz; must be > 0 |
| `observation_history_length` | int | `0` | Forwarded to `PreprocessorCore::set_config()` |
| `action_history_length` | int | `0` | Forwarded to `PreprocessorCore::set_config()` |

An empty plugin name or a non-positive `update_rate` makes `init()` throw `std::runtime_error`,
which the node's `main()` catches, logs at `FATAL`, and exits with code 1.

## Requires plugins to run

This package ships no concrete plugin implementations. Before the node can start, four plugin
packages implementing `ros2_policy_execution_core::{PreprocessorCore,InferenceCore,
PostprocessorCore,ExecutorCore}` must be built and exported via `pluginlib_export_plugin_description_file`.
Without them, `init()` throws a `pluginlib::PluginlibException` (e.g. `LibraryLoadException`)
naming the missing plugin, and the node exits with code 1 instead of starting.

## Example configuration

```yaml
policy_execution_pipeline:
  ros__parameters:
    preprocessor_plugin: "my_pkg/MyPreprocessor"
    inference_plugin: "my_pkg/MyOnnxInference"
    postprocessor_plugin: "my_pkg/MyPostprocessor"
    executor_plugin: "my_pkg/MyExecutor"
    update_rate: 100.0
    observation_history_length: 10
    action_history_length: 5
```

## Running

```bash
ros2 run ros2_policy_execution_pipeline policy_execution_pipeline_node --ros-args --params-file config.yaml
```
