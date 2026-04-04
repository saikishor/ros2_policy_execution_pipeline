// Copyright 2026 ros2_control Development Team
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
// Authors: Julia Jia

#include <array>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "ros2_policy_execution_adapters/ort_tensor_conversion.hpp"
#include "ros2_policy_execution_core/tensor/tensor_types.hpp"

namespace ros2_policy_execution_adapters
{

using ros2_policy_execution_core::DataType;
using ros2_policy_execution_core::ByteBufferView;
using ros2_policy_execution_core::Tensor;

TEST(OrtTensorConversionTest, TensorToOrtValueReusesPayloadBuffer)
{
  auto values = std::make_shared<std::vector<float>>(
    std::initializer_list<float>{1.0f, 2.0f, 3.0f, 4.0f});
  auto tensor = Tensor::share_vector(values, {2, 2});

  auto ort_reference = make_ort_value_reference(tensor);

  ASSERT_TRUE(ort_reference.value.IsTensor());
  EXPECT_EQ(ort_reference.value.GetTensorRawData(), tensor.raw_data());
  EXPECT_EQ(ort_reference.owner, tensor.buffer().owner());

  const auto shape_info = ort_reference.value.GetTensorTypeAndShapeInfo();
  EXPECT_EQ(shape_info.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
}

TEST(OrtTensorConversionTest, TensorOrtRoundTripPreservesShapeAndValuesWithoutCopy)
{
  auto values = std::make_shared<std::vector<float>>(
    std::initializer_list<float>{1.0f, 2.0f, 3.0f, 4.0f});
  auto tensor = Tensor::share_vector(values, {2, 2});

  auto ort_reference = make_ort_value_reference(tensor);
  auto round_trip = tensor_from_ort_value(std::move(ort_reference));

  EXPECT_EQ(round_trip.raw_data(), tensor.raw_data());
  EXPECT_EQ(round_trip.shape(), tensor.shape());
  EXPECT_EQ(round_trip.data_type(), DataType::Float32);

  auto span = round_trip.span<float>();
  ASSERT_EQ(span.size(), 4u);
  EXPECT_FLOAT_EQ(span[0], 1.0f);
  EXPECT_FLOAT_EQ(span[3], 4.0f);
}

TEST(OrtTensorConversionTest, BorrowTensorFromOrtValueCreatesView)
{
  std::vector<int64_t> data = {7, 8, 9};
  std::vector<int64_t> shape = {3};
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto ort_value = Ort::Value::CreateTensor<int64_t>(
    memory_info,
    data.data(),
    data.size(),
    shape.data(),
    shape.size());

  auto tensor = borrow_tensor_from_ort_value(ort_value);

  EXPECT_EQ(tensor.raw_data(), static_cast<const void *>(data.data()));
  EXPECT_EQ(tensor.data_type(), DataType::Int64);
  ASSERT_EQ(tensor.shape().size(), 1u);
  EXPECT_EQ(tensor.shape()[0], 3);

  auto span = tensor.span<int64_t>();
  EXPECT_EQ(span[0], 7);
  data[1] = 99;
  EXPECT_EQ(span[1], 99);
}

TEST(OrtTensorConversionTest, BorrowTensorFromConstOrtValueCreatesReadOnlyView)
{
  std::vector<int64_t> data = {1, 2};
  std::vector<int64_t> shape = {2};
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto ort_value = Ort::Value::CreateTensor<int64_t>(
    memory_info,
    data.data(),
    data.size(),
    shape.data(),
    shape.size());

  const Ort::Value & ort_value_const = ort_value;
  auto tensor = borrow_tensor_from_ort_value(ort_value_const);

  EXPECT_FALSE(tensor.buffer().is_mutable());
  EXPECT_EQ(tensor.raw_data(), static_cast<const void *>(data.data()));
  const auto & tensor_const = tensor;
  const auto span = tensor_const.span<int64_t>();
  EXPECT_EQ(span[0], 1);
}

TEST(OrtTensorConversionTest, ImmutableTensorCannotBeWrappedForOrtZeroCopy)
{
  const std::array<float, 3> values = {1.0f, 2.0f, 3.0f};
  Tensor tensor(
    DataType::Float32,
    {3},
    ByteBufferView::borrow(values.data(), values.size()));

  EXPECT_THROW(make_ort_value_reference(tensor), std::invalid_argument);
}

}  // namespace ros2_policy_execution_adapters
