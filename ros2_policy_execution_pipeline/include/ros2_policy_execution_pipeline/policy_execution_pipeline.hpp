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

#ifndef ROS2_POLICY_EXECUTION_PIPELINE__POLICY_EXECUTION_PIPELINE_HPP_
#define ROS2_POLICY_EXECUTION_PIPELINE__POLICY_EXECUTION_PIPELINE_HPP_

#include <memory>
#include <vector>

#include "pluginlib/class_loader.hpp"
#include "rclcpp/node.hpp"

#include "ros2_policy_execution_core/executor_core.hpp"
#include "ros2_policy_execution_core/inference_core.hpp"
#include "ros2_policy_execution_core/onnxruntime_types.hpp"
#include "ros2_policy_execution_core/postprocessor_core.hpp"
#include "ros2_policy_execution_core/preprocessor_core.hpp"

namespace ros2_policy_execution_pipeline
{

/**
 * @brief Loads the four pluginlib-based pipeline stages and drives them on a timer.
 *
 * Composition, not inheritance: this class holds an rclcpp::Node::SharedPtr rather than
 * being a node itself, since all four core classes take an rclcpp::Node::SharedPtr in their
 * configure() method. This sidesteps the two-phase-construction hazard of calling
 * shared_from_this() before the node is owned by a shared_ptr.
 */
class PolicyExecutionPipeline
{
public:
  /**
   * @brief Construct the pipeline around an existing ROS2 node.
   *
   * @param[in] node Shared pointer to the ROS2 node used for parameters, logging and the timer.
   */
  explicit PolicyExecutionPipeline(rclcpp::Node::SharedPtr node);

  /**
   * @brief Declare and read parameters, load and configure the four plugins, and start the timer.
   *
   * Must be called after construction and before rclcpp::spin(). Loads one plugin per pipeline
   * stage (preprocessor, inference, postprocessor, executor) using pluginlib, configures each of
   * them with the node, and creates a wall timer that calls update() at the configured rate.
   *
   * @throws std::runtime_error if a required parameter is missing or invalid (e.g. an empty
   *  plugin name or a non-positive update rate).
   * @throws pluginlib::PluginlibException if a plugin cannot be loaded or instantiated.
   */
  void init();

  /**
   * @brief Run a single pipeline cycle: preprocess, infer, postprocess, execute.
   *
   * Public so a single cycle can be driven manually (e.g. from tests) instead of by the timer.
   * Any exception raised by a pipeline stage is caught, logged (throttled) and swallowed, so a
   * single faulty cycle does not bring down the node.
   */
  void update();

private:
  rclcpp::Node::SharedPtr node_;

  // Loaders must be declared before the instances they create: members are destroyed in
  // reverse declaration order, so the instances are destroyed first and the loaders (which own
  // the dlopen'd plugin libraries) last. Reversing this order would unload a plugin's library
  // while an instance created from it is still alive.
  std::unique_ptr<pluginlib::ClassLoader<ros2_policy_execution_core::PreprocessorCore>>
  preprocessor_loader_;
  std::unique_ptr<pluginlib::ClassLoader<ros2_policy_execution_core::InferenceCore>>
  inference_loader_;
  std::unique_ptr<pluginlib::ClassLoader<ros2_policy_execution_core::PostprocessorCore>>
  postprocessor_loader_;
  std::unique_ptr<pluginlib::ClassLoader<ros2_policy_execution_core::ExecutorCore>>
  executor_loader_;

  std::shared_ptr<ros2_policy_execution_core::PreprocessorCore> preprocessor_;
  std::shared_ptr<ros2_policy_execution_core::InferenceCore> inference_;
  std::shared_ptr<ros2_policy_execution_core::PostprocessorCore> postprocessor_;
  std::shared_ptr<ros2_policy_execution_core::ExecutorCore> executor_;

  rclcpp::TimerBase::SharedPtr timer_;

  /// Reused across cycles to avoid reallocating the output vector every update().
  std::vector<ros2_policy_execution_core::OrtValueSharedPtr> inference_output_;
};

}  // namespace ros2_policy_execution_pipeline

#endif  // ROS2_POLICY_EXECUTION_PIPELINE__POLICY_EXECUTION_PIPELINE_HPP_
