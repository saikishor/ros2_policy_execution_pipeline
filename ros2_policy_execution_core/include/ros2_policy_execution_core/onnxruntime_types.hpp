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

#ifndef ROS2_POLICY_EXECUTION_CORE__ONNXRUNTIME_TYPES_HPP_
#define ROS2_POLICY_EXECUTION_CORE__ONNXRUNTIME_TYPES_HPP_

#include <onnxruntime_cxx_api.h>

#include <memory>

namespace ros2_policy_execution_core
{

/// Shared pointer alias for Ort::Value, following the rclcpp SharedPtr convention.
using OrtValueSharedPtr = std::shared_ptr<Ort::Value>;

}  // namespace ros2_policy_execution_core

#endif  // ROS2_POLICY_EXECUTION_CORE__ONNXRUNTIME_TYPES_HPP_
