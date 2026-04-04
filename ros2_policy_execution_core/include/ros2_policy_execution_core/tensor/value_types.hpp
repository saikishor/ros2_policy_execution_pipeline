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

#ifndef ROS2_POLICY_EXECUTION_CORE__TENSOR__VALUE_TYPES_HPP_
#define ROS2_POLICY_EXECUTION_CORE__TENSOR__VALUE_TYPES_HPP_

/**
 * @file value_types.hpp
 * @brief Aggregate header for tensor, device, and named-value types in the policy pipeline.
 *
 * Include only the headers required for the symbols in use, in order to keep compile-time dependencies small:
 * - tensor/data_type.hpp — DataType, DataTypeForElement, data_type_v, data_type_size
 * - tensor/tensor_device.hpp — Device, DeviceType
 * - tensor/tensor_types.hpp — ByteBufferView, Tensor (includes data_type.hpp)
 * - tensor/named_value_list.hpp — Value, NamedValueList, find_value
 *
 * \sa doc/developer_guide.md
 */

#include "ros2_policy_execution_core/tensor/tensor_device.hpp"
#include "ros2_policy_execution_core/tensor/tensor_types.hpp"
#include "ros2_policy_execution_core/tensor/named_value_list.hpp"

#endif  // ROS2_POLICY_EXECUTION_CORE__TENSOR__VALUE_TYPES_HPP_
