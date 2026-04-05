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

#ifndef ROS2_POLICY_EXECUTION_CORE__TENSOR__NAMED_VALUE_LIST_HPP_
#define ROS2_POLICY_EXECUTION_CORE__TENSOR__NAMED_VALUE_LIST_HPP_

/**
 * @file named_value_list.hpp
 * @brief Named values for ordered model inputs and outputs.
 */

#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "ros2_policy_execution_core/tensor/tensor_types.hpp"

namespace ros2_policy_execution_core
{

/**
 * @brief Discriminated value exchanged between pipeline stages.
 */
class Value
{
public:
  /// @brief Discriminator for the active payload.
  enum class Kind
  {
    kEmpty = 0,  ///< No payload.
    kTensor      ///< Dense tensor payload.
  };

  /// @brief Constructs an empty value (Kind::kEmpty).
  Value() = default;

  /// @brief Constructs a value holding the given tensor.
  explicit Value(Tensor tensor)
  : payload_(std::move(tensor))
  {}

  /// @brief Active kind of the stored payload.
  [[nodiscard]] Kind kind() const
  {
    if (std::holds_alternative<Tensor>(payload_)) {
      return Kind::kTensor;
    }
    return Kind::kEmpty;
  }

  /// @brief Whether the active payload is a tensor.
  [[nodiscard]] bool is_tensor() const
  {
    return std::holds_alternative<Tensor>(payload_);
  }

  /// @brief Const tensor payload; valid only if is_tensor() is true.
  [[nodiscard]] const Tensor & as_tensor() const
  {
    return std::get<Tensor>(payload_);
  }

  /// @brief Mutable tensor payload; valid only if is_tensor() is true.
  [[nodiscard]] Tensor & as_tensor()
  {
    return std::get<Tensor>(payload_);
  }

private:
  std::variant<std::monostate, Tensor> payload_ = {};
};

/**
 * @brief One named entry in an ordered model input or output list.
 */
struct NamedValue
{
  std::string name;  //!< Model input or output identifier.
  Value value;       //!< Payload associated with \p name.
};

//! Ordered sequence of named values; container order defines model I/O ordering.
using NamedValueList = std::vector<NamedValue>;

/**
 * @brief Find the first entry matching \p name; returns nullptr if not found.
 * @param[in] values Sequence to search.
 * @param[in] name Name to match.
 */
inline const Value * find_value(const NamedValueList & values, const std::string & name)
{
  for (const auto & entry : values) {
    if (entry.name == name) {
      return &entry.value;
    }
  }
  return nullptr;
}

/// @brief Mutable overload of find_value.
inline Value * find_value(NamedValueList & values, const std::string & name)
{
  for (auto & entry : values) {
    if (entry.name == name) {
      return &entry.value;
    }
  }
  return nullptr;
}

}  // namespace ros2_policy_execution_core

#endif  // ROS2_POLICY_EXECUTION_CORE__TENSOR__NAMED_VALUE_LIST_HPP_
