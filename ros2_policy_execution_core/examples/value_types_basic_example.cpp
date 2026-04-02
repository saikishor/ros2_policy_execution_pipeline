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

#include <iostream>
#include <memory>
#include <vector>

#include "ros2_policy_execution_core/ort_value_conversion.hpp"
#include "ros2_policy_execution_core/value_types.hpp"

namespace
{

int run_example()
{
  using ros2_policy_execution_core::Tensor;
  using ros2_policy_execution_core::Value;
  using ros2_policy_execution_core::ValueSet;
  using ros2_policy_execution_core::make_ort_value_reference;
  using ros2_policy_execution_core::tensor_from_ort_value;

  // 1. Create the payload once in a shared vector.
  auto observations = std::make_shared<std::vector<float>>(
    std::initializer_list<float>{1.0f, 2.0f, 3.0f, 4.0f});

  // 2. Wrap that same storage as an ordered named tensor input.
  ValueSet inputs = {
    {"observation", Value(Tensor::share_vector(observations, {1, 4}))}
  };

  auto & tensor = inputs.front().value.as_tensor();

  // 3. Create an Ort::Value view over the same payload bytes.
  auto ort_reference = make_ort_value_reference(tensor);

  std::cout << "Tensor payload pointer: " << tensor.raw_data() << '\n';
  std::cout << "Ort::Value pointer:      " << ort_reference.value.GetTensorRawData() << '\n';
  std::cout << "Zero-copy wrapping:      "
            << std::boolalpha
            << (tensor.raw_data() == ort_reference.value.GetTensorRawData())
            << '\n';

  // 4. Convert back from the ONNX Runtime representation to the generic tensor
  //    type and prove that the values still come from the same storage.
  auto round_trip = tensor_from_ort_value(std::move(ort_reference));
  observations.reset();

  std::cout << "Round-trip values:";
  for (const auto value : round_trip.span<float>()) {
    std::cout << ' ' << value;
  }
  std::cout << '\n';

  return 0;
}

}  // namespace

int main()
{
  return run_example();
}
