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

#include <vector>

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
   * @brief Process the inference output and return the final commands.
   *
   * This pure virtual method must be implemented by derived classes to perform
   * the final postprocessing on the inference output and produce the commands to be sent.
   *
   * @param[in] inference_output The output vector from inference to be postprocessed.
   * @return The final commands vector to be sent.
   */
  virtual [[nodiscard]] const std::vector<float> & process(const std::vector<float> & inference_output) = 0;
};

}  // namespace ros2_policy_execution_core

#endif  // ROS2_POLICY_EXECUTION_CORE__POSTPROCESSOR_CORE_HPP_
