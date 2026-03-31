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

#include <vector>

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
   * @brief Run inference on the given observations and produce outputs.
   *
   * This pure virtual method must be implemented by derived classes to perform
   * the actual inference computation using the loaded policy model.
   *
   * @param[in] obs The observation vector to use as input for inference.
   * @param[out] output The output vector that will be populated with the inference results.
   * @return true if inference was successful, false otherwise.
   */
  virtual bool run_inference(const std::vector<float> & obs, std::vector<float> & output) = 0;
};

}  // namespace ros2_policy_execution_core

#endif  // ROS2_POLICY_EXECUTION_CORE__INFERENCE_CORE_HPP_
