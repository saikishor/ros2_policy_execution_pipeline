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

#ifndef ROS2_POLICY_EXECUTION_CORE__POSTPROCESSOR_CORE_HPP_
#define ROS2_POLICY_EXECUTION_CORE__POSTPROCESSOR_CORE_HPP_

#include "rclcpp/node.hpp"

#include "ros2_policy_execution_core/tensor/named_value_list.hpp"

namespace ros2_policy_execution_core
{

/**
 * @brief Abstract base class for postprocessors in the policy execution pipeline.
 *
 * This class serves as a plugin base class for creating custom postprocessors
 * that transform raw output from inference into final commands to be sent.
 */
class PostprocessorCore
{
public:
  /**
   * @brief Virtual destructor for proper cleanup of derived classes.
   */
  virtual ~PostprocessorCore() = default;

  /**
   * @brief Configure the postprocessor with necessary parameters or ROS2 node.
   *
   * This pure virtual method must be implemented by derived classes to perform
   * any necessary configuration steps, such as reading parameters from the ROS2 node.
   * Creating appropriate subscriptions or service clients can also be done here.
   *
   * @param[in] node Shared pointer to the ROS2 node for accessing parameters and other resources.
   */
  virtual void configure(const rclcpp::Node::SharedPtr & node) = 0;

  /**
   * @brief Process inference outputs and return final command tensors (or other values).
   *
   * @param[in] inference_output Values produced by `InferenceCore::run_inference`.
   * @return Postprocessed values (for example a float tensor of joint commands).
   */
  [[nodiscard]] virtual NamedValueList process(const NamedValueList & inference_output) = 0;
};

}  // namespace ros2_policy_execution_core

#endif  // ROS2_POLICY_EXECUTION_CORE__POSTPROCESSOR_CORE_HPP_
