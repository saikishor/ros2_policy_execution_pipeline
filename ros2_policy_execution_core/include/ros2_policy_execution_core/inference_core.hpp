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

#ifndef ROS2_POLICY_EXECUTION_CORE__INFERENCE_CORE_HPP_
#define ROS2_POLICY_EXECUTION_CORE__INFERENCE_CORE_HPP_

#include "rclcpp/node.hpp"

#include "ros2_policy_execution_core/tensor/named_value_list.hpp"

namespace ros2_policy_execution_core
{

/**
 * @brief Abstract base class for inference engines in the policy execution pipeline.
 *
 * This class serves as a plugin base class for creating custom inference engines
 * that run policy inference given observations and produce outputs.
 */
class InferenceCore
{
public:
  /**
   * @brief Virtual destructor for proper cleanup of derived classes.
   */
  virtual ~InferenceCore() = default;

  /**
   * @brief Configure the inference engine with necessary parameters or ROS2 node.
   *
   * This pure virtual method must be implemented by derived classes to perform
   * any necessary configuration steps, such as loading policy models or reading
   * parameters from the ROS2 node.
   *
   * @param[in] node Shared pointer to the ROS2 node for accessing parameters and other resources.
   */
  virtual void configure(const rclcpp::Node::SharedPtr & node) = 0;

  /**
   * @brief Run inference on named tensor inputs and populate named outputs.
   *
   * The default preprocessor supplies one tensor, usually named `observation`
   * (`PreprocessorCore::get_observation_named_value_list()`). Models may require other names.
   *
   * @param[in] inputs Ordered named tensors (and future value kinds) for the model.
   * @param[out] outputs Populated by the implementation on success.
   * @return true if inference was successful, false otherwise.
   */
  virtual bool run_inference(const NamedValueList & inputs, NamedValueList & outputs) = 0;
};

}  // namespace ros2_policy_execution_core

#endif  // ROS2_POLICY_EXECUTION_CORE__INFERENCE_CORE_HPP_
