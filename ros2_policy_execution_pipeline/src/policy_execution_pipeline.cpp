// Copyright 2026 PAL Robotics S.L.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/// \author Sai Kishor Kothakota

#include "ros2_policy_execution_pipeline/policy_execution_pipeline.hpp"

#include <chrono>
#include <stdexcept>
#include <string>

using ros2_policy_execution_core::ExecutorCore;
using ros2_policy_execution_core::InferenceCore;
using ros2_policy_execution_core::PostprocessorCore;
using ros2_policy_execution_core::PreprocessorCore;
using ros2_policy_execution_core::PreprocessorCoreConfig;

namespace
{
constexpr char kPreprocessorBaseClass[] = "ros2_policy_execution_core::PreprocessorCore";
constexpr char kInferenceBaseClass[] = "ros2_policy_execution_core::InferenceCore";
constexpr char kPostprocessorBaseClass[] = "ros2_policy_execution_core::PostprocessorCore";
constexpr char kExecutorBaseClass[] = "ros2_policy_execution_core::ExecutorCore";
constexpr char kBasePackage[] = "ros2_policy_execution_core";

/// Reads a required, non-empty string parameter, declaring it first.
/// @throws std::runtime_error if the parameter is left empty.
std::string declare_and_get_required_string(
  const rclcpp::Node::SharedPtr & node, const std::string & name)
{
  node->declare_parameter<std::string>(name, "");
  const auto value = node->get_parameter(name).as_string();
  if (value.empty()) {
    throw std::runtime_error("Required parameter '" + name + "' was not set.");
  }
  return value;
}
}  // namespace

namespace ros2_policy_execution_pipeline
{

PolicyExecutionPipeline::PolicyExecutionPipeline(rclcpp::Node::SharedPtr node)
: node_(std::move(node))
{
}

void PolicyExecutionPipeline::init()
{
  const auto preprocessor_plugin = declare_and_get_required_string(node_, "preprocessor_plugin");
  const auto inference_plugin = declare_and_get_required_string(node_, "inference_plugin");
  const auto postprocessor_plugin = declare_and_get_required_string(node_, "postprocessor_plugin");
  const auto executor_plugin = declare_and_get_required_string(node_, "executor_plugin");

  node_->declare_parameter<double>("update_rate", 100.0);
  const double update_rate = node_->get_parameter("update_rate").as_double();
  if (update_rate <= 0.0) {
    throw std::runtime_error("Parameter 'update_rate' must be > 0, got " +
            std::to_string(update_rate) + ".");
  }

  node_->declare_parameter<int>("observation_history_length", 0);
  node_->declare_parameter<int>("action_history_length", 0);
  PreprocessorCoreConfig history_config;
  history_config.observation_history_length = static_cast<size_t>(
    node_->get_parameter("observation_history_length").as_int());
  history_config.action_history_length = static_cast<size_t>(
    node_->get_parameter("action_history_length").as_int());

  preprocessor_loader_ =
    std::make_unique<pluginlib::ClassLoader<PreprocessorCore>>(
    kBasePackage, kPreprocessorBaseClass);
  inference_loader_ =
    std::make_unique<pluginlib::ClassLoader<InferenceCore>>(kBasePackage, kInferenceBaseClass);
  postprocessor_loader_ =
    std::make_unique<pluginlib::ClassLoader<PostprocessorCore>>(
    kBasePackage, kPostprocessorBaseClass);
  executor_loader_ =
    std::make_unique<pluginlib::ClassLoader<ExecutorCore>>(kBasePackage, kExecutorBaseClass);

  preprocessor_ = preprocessor_loader_->createSharedInstance(preprocessor_plugin);
  inference_ = inference_loader_->createSharedInstance(inference_plugin);
  postprocessor_ = postprocessor_loader_->createSharedInstance(postprocessor_plugin);
  executor_ = executor_loader_->createSharedInstance(executor_plugin);

  // Set the history configuration before configure(), so a derived configure() implementation
  // can already observe the configured history lengths if it needs to.
  preprocessor_->set_config(history_config);

  preprocessor_->configure(node_);
  inference_->configure(node_);
  postprocessor_->configure(node_);
  executor_->configure(node_);

  RCLCPP_INFO(
    node_->get_logger(),
    "Policy execution pipeline configured: preprocessor='%s', inference='%s', "
    "postprocessor='%s', executor='%s', update_rate=%.2f Hz",
    preprocessor_plugin.c_str(), inference_plugin.c_str(), postprocessor_plugin.c_str(),
    executor_plugin.c_str(), update_rate);

  timer_ = node_->create_wall_timer(
    std::chrono::duration<double>(1.0 / update_rate),
    std::bind(&PolicyExecutionPipeline::update, this));
}

void PolicyExecutionPipeline::update()
{
  try {
    const rclcpp::Time now = node_->get_clock()->now();

    if (!preprocessor_->build_observation(now)) {
      RCLCPP_WARN_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 1000,
        "Failed to build observation, skipping this cycle.");
      return;
    }
    const auto & observation = preprocessor_->get_observation();

    inference_output_.clear();
    if (!inference_->run_inference(observation, inference_output_)) {
      RCLCPP_ERROR_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 1000,
        "Inference failed, skipping this cycle.");
      return;
    }

    const auto & commands = postprocessor_->process(inference_output_);

    if (!executor_->execute(commands)) {
      RCLCPP_ERROR_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 1000,
        "Executor failed to send commands.");
    }

    // Only record history for a cycle that made it all the way through observation building
    // and inference, so a failed cycle does not poison the history buffers with partial data.
    preprocessor_->set_previous_observations(observation);
    preprocessor_->set_previous_actions(inference_output_);
  } catch (const std::exception & e) {
    // PreprocessorCore::build_observation() signals invalid provider data by throwing
    // std::runtime_error rather than returning false, so this catch is load-bearing: without
    // it, a single stale observation timestamp would escape the timer callback and crash the
    // node instead of just skipping a cycle.
    RCLCPP_ERROR_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 1000,
      "Pipeline cycle failed: %s", e.what());
  }
}

}  // namespace ros2_policy_execution_pipeline
