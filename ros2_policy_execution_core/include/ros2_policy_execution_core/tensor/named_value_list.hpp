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
 * @brief Named values for ordered model inputs and outputs (dense tensors in the current API).
 *
 * Includes `tensor_types.hpp`. Include only the headers required for the symbols in use
 * in order to keep compile-time dependencies small; `tensor/data_type.hpp`,
 * `tensor/tensor_device.hpp`, or `tensor/tensor_types.hpp` alone suffice when named lists are not required.
 */

#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "ros2_policy_execution_core/tensor/tensor_types.hpp"

namespace ros2_policy_execution_core
{

/**
 * @brief Discriminated value exchanged between preprocessors, inference backends, and postprocessors.
 *
 * The current representation supports dense tensors only; further std::variant alternatives
 * may be added without changing the Value class interface.
 */
class Value
{
public:
  /**
   * @brief Discriminator for the active payload.
   */
  enum class Kind
  {
    kEmpty = 0,  ///< No payload.
    kTensor      ///< Dense tensor payload.
  };

  /// @brief Constructs an empty value (Kind::kEmpty).
  Value() = default;

  /**
   * @brief Constructs a value whose active payload is the given tensor.
   * @param[in] tensor Tensor to store.
   */
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
  std::string name;  //!< Model input or output identifier (for example, an ONNX binding name).
  Value value;       //!< Payload associated with \p name.
};

//! Ordered sequence of named values; container order defines model I/O ordering.
using NamedValueList = std::vector<NamedValue>;

/**
 * @brief Linear search for the first entry whose name equals \p name.
 * @param[in] values Sequence to search.
 * @param[in] name Name to match.
 * @return Pointer to the matching Value, or nullptr if not found.
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

/**
 * @brief Mutable overload; behavior matches the overload for const NamedValueList.
 * @param[in] values Sequence to search.
 * @param[in] name Name to match.
 * @return Pointer to the matching Value, or nullptr if not found.
 */
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
