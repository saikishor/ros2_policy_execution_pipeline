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

#include <array>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "ros2_policy_execution_core/ort_value_conversion.hpp"

namespace ros2_policy_execution_core
{

TEST(OrtValueConversionTest, TensorToOrtValueReusesPayloadBuffer)
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
  EXPECT_EQ(shape_info.GetShape(), std::vector<int64_t>({2, 2}));
}

TEST(OrtValueConversionTest, TensorOrtRoundTripPreservesShapeAndValuesWithoutCopy)
{
  auto values = std::make_shared<std::vector<float>>(
    std::initializer_list<float>{1.0f, 2.0f, 3.0f, 4.0f});
  auto tensor = Tensor::share_vector(values, {2, 2});

  auto ort_reference = make_ort_value_reference(tensor);
  auto round_trip = tensor_from_ort_value(std::move(ort_reference));
  values.reset();

  EXPECT_EQ(round_trip.data_type(), DataType::kFloat32);
  EXPECT_EQ(round_trip.shape(), std::vector<int64_t>({2, 2}));
  EXPECT_EQ(round_trip.raw_data(), tensor.raw_data());
  EXPECT_TRUE(round_trip.buffer().is_mutable());

  auto span = round_trip.span<float>();
  ASSERT_EQ(span.size(), 4u);
  EXPECT_FLOAT_EQ(span[0], 1.0f);
  EXPECT_FLOAT_EQ(span[1], 2.0f);
  EXPECT_FLOAT_EQ(span[2], 3.0f);
  EXPECT_FLOAT_EQ(span[3], 4.0f);
  span[1] = 9.0f;
  EXPECT_FLOAT_EQ(span[1], 9.0f);
}

TEST(OrtValueConversionTest, BorrowTensorFromOrtValueCreatesView)
{
  std::vector<int64_t> values = {5, 6, 7};
  std::vector<int64_t> shape = {3};
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto ort_value = Ort::Value::CreateTensor<int64_t>(
    memory_info,
    values.data(),
    values.size(),
    shape.data(),
    shape.size());

  auto tensor = borrow_tensor_from_ort_value(ort_value);

  EXPECT_EQ(tensor.data_type(), DataType::kInt64);
  EXPECT_EQ(tensor.shape(), std::vector<int64_t>({3}));
  EXPECT_EQ(tensor.raw_data(), values.data());
  EXPECT_TRUE(tensor.buffer().is_mutable());

  auto span = tensor.span<int64_t>();
  ASSERT_EQ(span.size(), 3u);
  EXPECT_EQ(span[0], 5);
  EXPECT_EQ(span[1], 6);
  EXPECT_EQ(span[2], 7);
  span[0] = 10;
  EXPECT_EQ(values[0], 10);
}

TEST(OrtValueConversionTest, BorrowTensorFromConstOrtValueCreatesReadOnlyView)
{
  std::vector<int64_t> values = {5, 6, 7};
  std::vector<int64_t> shape = {3};
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto ort_value = Ort::Value::CreateTensor<int64_t>(
    memory_info,
    values.data(),
    values.size(),
    shape.data(),
    shape.size());

  const Ort::Value & ort_value_const = ort_value;
  auto tensor = borrow_tensor_from_ort_value(ort_value_const);

  EXPECT_FALSE(tensor.buffer().is_mutable());
  const Tensor & tensor_const = tensor;
  const auto span = tensor_const.span<int64_t>();
  ASSERT_EQ(span.size(), 3u);
  EXPECT_EQ(span[0], 5);
  EXPECT_THROW(
    static_cast<void>(tensor.span<int64_t>()),
    std::runtime_error);
}

TEST(OrtValueConversionTest, ImmutableTensorCannotBeWrappedForOrtZeroCopy)
{
  const std::array<float, 2> values = {1.0f, 2.0f};
  Tensor tensor(
    DataType::kFloat32,
    {2},
    SharedBuffer::borrow(values.data(), values.size()));

  EXPECT_THROW(make_ort_value_reference(tensor), std::invalid_argument);
}

}  // namespace ros2_policy_execution_core
