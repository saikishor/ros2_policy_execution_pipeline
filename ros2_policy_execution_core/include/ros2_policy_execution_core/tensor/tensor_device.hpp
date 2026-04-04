// Copyright 2026 PAI SIG
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
//
// Authors: Jennifer Buehler, Julia Jia

#ifndef ROS2_POLICY_EXECUTION_CORE__TENSOR__TENSOR_DEVICE_HPP_
#define ROS2_POLICY_EXECUTION_CORE__TENSOR__TENSOR_DEVICE_HPP_

/**
 * @file tensor_device.hpp
 * @brief Device classification and lightweight metadata for tensors.
 *
 * \sa doc/developer_guide.md
 */

namespace ros2_policy_execution_core
{

/**
 * @brief Logical device family for where tensor data resides (CPU, CUDA, or custom).
 */
enum class DeviceType
{
  Cpu = 0,   ///< Host CPU.
  Cuda,      ///< NVIDIA CUDA device.
  Custom     ///< Vendor- or application-defined device.
};

/**
 * @brief Device placement metadata carried alongside tensor views.
 */
struct Device
{
  DeviceType type = DeviceType::Cpu;  //!< Device class for tensor storage; see DeviceType.
  int device_id = 0;  //!< For Cuda: device ordinal. Ignored for Cpu; meaning is Custom-specific.
};

}  // namespace ros2_policy_execution_core

#endif  // ROS2_POLICY_EXECUTION_CORE__TENSOR__TENSOR_DEVICE_HPP_
