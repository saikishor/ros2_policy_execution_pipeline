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

#ifndef ROS2_POLICY_EXECUTION_CORE__TENSOR__DATA_TYPE_HPP_
#define ROS2_POLICY_EXECUTION_CORE__TENSOR__DATA_TYPE_HPP_

/**
 * @file data_type.hpp
 * @brief Pipeline element datatype enum and C++ scalar mapping traits.
 *
 * For buffer and tensor views, include `tensor/tensor_types.hpp` (which includes this header).
 */

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ros2_policy_execution_core
{

/**
 * @brief Supported element types for dense tensors in the policy pipeline.
 */
enum class DataType
{
  Unknown = 0,
  Float32,
  Float64,
  Int32,
  Int64,
  UInt8,
  Bool
};

/**
 * @brief Size in bytes of one element of \p data_type; 0 for Unknown.
 * @param[in] data_type Datatype to query.
 */
[[nodiscard]] constexpr std::size_t data_type_size(const DataType data_type) noexcept
{
  switch (data_type) {
    case DataType::Float32:
      return sizeof(float);
    case DataType::Float64:
      return sizeof(double);
    case DataType::Int32:
      return sizeof(int32_t);
    case DataType::Int64:
      return sizeof(int64_t);
    case DataType::UInt8:
      return sizeof(uint8_t);
    case DataType::Bool:
      return sizeof(bool);
    case DataType::Unknown:
      return 0;
    default:
      return 0;
  }
}

/**
 * @brief Maps a C++ scalar element type to \ref DataType; unmapped types yield DataType::Unknown.
 */
template<typename T>
struct DataTypeForElement
{
  static constexpr DataType value = DataType::Unknown;
};

template<>
struct DataTypeForElement<float>
{
  static constexpr DataType value = DataType::Float32;
};

template<>
struct DataTypeForElement<double>
{
  static constexpr DataType value = DataType::Float64;
};

template<>
struct DataTypeForElement<int32_t>
{
  static constexpr DataType value = DataType::Int32;
};

template<>
struct DataTypeForElement<int64_t>
{
  static constexpr DataType value = DataType::Int64;
};

template<>
struct DataTypeForElement<uint8_t>
{
  static constexpr DataType value = DataType::UInt8;
};

template<>
struct DataTypeForElement<bool>
{
  static constexpr DataType value = DataType::Bool;
};

/**
 * @brief Compile-time \ref DataType for C++ element type \p T (cv-qualified \p T uses the unqualified mapping).
 * @tparam T C++ element type to map.
 */
template<typename T>
inline constexpr DataType data_type_v = DataTypeForElement<std::remove_cv_t<T>>::value;

}  // namespace ros2_policy_execution_core

#endif  // ROS2_POLICY_EXECUTION_CORE__TENSOR__DATA_TYPE_HPP_
