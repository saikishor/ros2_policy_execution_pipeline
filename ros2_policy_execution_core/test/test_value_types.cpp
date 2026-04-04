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
#include <string>
#include <vector>

#include "gtest/gtest.h"

// `NamedValueList` / `find_value` tests need this header; it pulls in `tensor/tensor_types.hpp`.
// Dtype-only: `tensor/data_type.hpp`. Tensor / ByteBufferView: `tensor/tensor_types.hpp`.
#include "ros2_policy_execution_core/tensor/named_value_list.hpp"

namespace ros2_policy_execution_core
{

TEST(ValueTypesTest, TensorSharesVectorStorageWithoutCopy)
{
  auto values = std::make_shared<std::vector<float>>(
    std::initializer_list<float>{1.0f, 2.0f, 3.0f, 4.0f});

  auto tensor = Tensor::share_vector(values, {2, 2});

  EXPECT_EQ(tensor.raw_data(), values->data());
  EXPECT_EQ(tensor.bytes(), values->size() * sizeof(float));
  EXPECT_EQ(tensor.rank(), 2u);
  EXPECT_EQ(tensor.num_elements(), 4u);

  auto span = tensor.span<float>();
  span[1] = 7.5f;
  EXPECT_FLOAT_EQ((*values)[1], 7.5f);
}

TEST(ValueTypesTest, TensorKeepsSharedStorageAlive)
{
  auto values = std::make_shared<std::vector<int64_t>>(
    std::initializer_list<int64_t>{10, 20, 30});

  auto tensor = Tensor::share_vector(values, {3});
  values.reset();

  const auto span = tensor.span<int64_t>();
  ASSERT_EQ(span.size(), 3u);
  EXPECT_EQ(span[0], 10);
  EXPECT_EQ(span[1], 20);
  EXPECT_EQ(span[2], 30);
}

TEST(ValueTypesTest, ByteOffsetCreatesSubviewWithoutCopy)
{
  auto values = std::make_shared<std::vector<float>>(
    std::initializer_list<float>{1.0f, 2.0f, 3.0f, 4.0f});

  Tensor tensor(
    DataType::Float32,
    {2},
    ByteBufferView::share_vector(values),
    {},
    {},
    2 * sizeof(float));

  const auto span = tensor.span<float>();
  ASSERT_EQ(span.size(), 2u);
  EXPECT_FLOAT_EQ(span[0], 3.0f);
  EXPECT_FLOAT_EQ(span[1], 4.0f);
  EXPECT_EQ(tensor.raw_data(), values->data() + 2);
}

TEST(ValueTypesTest, NamedValueListPreservesOrderAndLookup)
{
  auto image_values = std::make_shared<std::vector<uint8_t>>(
    std::initializer_list<uint8_t>{1, 2, 3});
  auto pose_values = std::make_shared<std::vector<double>>(
    std::initializer_list<double>{0.1, 0.2, 0.3});

  NamedValueList inputs = {
    {"image", Value(Tensor::share_vector(image_values, {3}))},
    {"pose", Value(Tensor::share_vector(pose_values, {3}))}
  };

  ASSERT_EQ(inputs.size(), 2u);
  EXPECT_EQ(inputs[0].name, "image");
  EXPECT_EQ(inputs[1].name, "pose");

  const auto * image_value = find_value(inputs, "image");
  ASSERT_NE(image_value, nullptr);
  ASSERT_TRUE(image_value->is_tensor());
  EXPECT_EQ(image_value->as_tensor().data_type(), DataType::UInt8);

  EXPECT_EQ(find_value(inputs, "missing"), nullptr);
}

TEST(ValueTypesTest, ImmutableBufferRejectsMutableAccess)
{
  const std::array<float, 3> values = {1.0f, 2.0f, 3.0f};
  Tensor tensor(
    DataType::Float32,
    {3},
    ByteBufferView::borrow(values.data(), values.size()));

  EXPECT_FALSE(tensor.buffer().is_mutable());
  EXPECT_THROW({static_cast<void>(tensor.mutable_raw_data());}, std::runtime_error);
  EXPECT_THROW(
    static_cast<void>(tensor.span<float>()),
    std::runtime_error);
}

TEST(ValueTypesTest, AvailableBytesClampsInvalidOffset)
{
  Tensor tensor;
  EXPECT_EQ(tensor.available_bytes(), 0u);
}

TEST(ValueTypesTest, DataTypeForElementMapsScalars)
{
  EXPECT_EQ(data_type_v<float>, DataType::Float32);
  EXPECT_EQ(data_type_v<double>, DataType::Float64);
  EXPECT_EQ(data_type_v<int32_t>, DataType::Int32);
  EXPECT_EQ(data_type_v<uint8_t>, DataType::UInt8);
}

TEST(ValueTypesTest, DataTypeSizeIsZeroForUnknown)
{
  EXPECT_EQ(data_type_size(DataType::Unknown), 0u);
  EXPECT_EQ(data_type_size(DataType::Float32), sizeof(float));
}

TEST(ValueTypesTest, ValueEmptyKindIsNotTensor)
{
  Value v;
  EXPECT_EQ(v.kind(), Value::Kind::kEmpty);
  EXPECT_FALSE(v.is_tensor());
}

TEST(ValueTypesTest, FindValueMutableOverloadReplacesEntry)
{
  NamedValueList list{{"x", Value()}};
  Value * pv = find_value(list, "x");
  ASSERT_NE(pv, nullptr);
  auto storage = std::make_shared<std::vector<float>>(std::vector<float>{9.0f});
  *pv = Value(Tensor::share_vector(storage, {1}));
  ASSERT_TRUE(list[0].value.is_tensor());
  EXPECT_FLOAT_EQ(list[0].value.as_tensor().span<float>()[0], 9.0f);
}

}  // namespace ros2_policy_execution_core
