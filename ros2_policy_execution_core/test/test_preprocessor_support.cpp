// Copyright (C) 2026 ros2_control Development Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Julia Jia

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT(build/include_subdir)
#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"

#include "ros2_policy_execution_core/preprocessor_support.hpp"

namespace ros2_policy_execution_core
{

namespace
{
/// Creates one 1-D single-element Ort::Value tensor per float, using ORT's allocator.
/// The returned shared_ptrs own their memory — no external backing data required.
std::vector<std::shared_ptr<Ort::Value>> make_ort_values(const std::vector<float> & data)
{
  static Ort::AllocatorWithDefaultOptions allocator;
  std::vector<int64_t> shape = {1};
  std::vector<std::shared_ptr<Ort::Value>> values;
  values.reserve(data.size());
  for (float v : data) {
    auto tensor = std::make_shared<Ort::Value>(
      Ort::Value::CreateTensor<float>(allocator, shape.data(), 1));
    *tensor->GetTensorMutableData<float>() = v;
    values.push_back(std::move(tensor));
  }
  return values;
}

/// Creates a single Ort::Value tensor of type T with the given shape and data.
template<typename T>
std::shared_ptr<Ort::Value> make_typed_tensor(
  const std::vector<T> & data,
  const std::vector<int64_t> & shape)
{
  static Ort::AllocatorWithDefaultOptions allocator;
  auto tensor = std::make_shared<Ort::Value>(
    Ort::Value::CreateTensor<T>(allocator, shape.data(), shape.size()));
  std::copy(data.begin(), data.end(), tensor->template GetTensorMutableData<T>());
  return tensor;
}

/// Returns the total element count for a shape vector.
int64_t shape_size(const std::vector<int64_t> & shape)
{
  return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
}
}  // namespace

class PreprocessorSupportTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    rclcpp::init(0, nullptr);
  }

  void TearDown() override
  {
    rclcpp::shutdown();
  }
};

TEST_F(PreprocessorSupportTest, ObservationProviderRegistry_RegisterOrderAndParallelSegments)
{
  ObservationProviderRegistry registry;
  ASSERT_TRUE(registry.empty());

  std::vector<float> raw1 = {1.0f};
  std::vector<float> raw2 = {2.0f, 3.0f};
  auto v1 = make_ort_values(raw1);
  auto v2 = make_ort_values(raw2);
  rclcpp::Time t1(1, 0);
  rclcpp::Time t2(2, 0);
  ObservationData d1{v1, t1};
  ObservationData d2{v2, t2};

  registry.register_provider("p1", {"a"}, [&d1]() -> const ObservationData & {return d1;});
  registry.register_provider("p2", {"b", "c"}, [&d2]() -> const ObservationData & {return d2;});

  EXPECT_FALSE(registry.empty());
  ASSERT_EQ(registry.providers().size(), 2u);
  ASSERT_EQ(registry.segment_names().size(), 2u);

  EXPECT_EQ(registry.providers()[0].first, "p1");
  EXPECT_EQ(registry.providers()[1].first, "p2");
  EXPECT_EQ(registry.segment_names()[0].first, "p1");
  EXPECT_EQ(registry.segment_names()[1].first, "p2");
  ASSERT_EQ(registry.segment_names()[0].second.size(), 1u);
  EXPECT_EQ(registry.segment_names()[0].second[0], "a");
  ASSERT_EQ(registry.segment_names()[1].second.size(), 2u);
  EXPECT_EQ(registry.segment_names()[1].second[0], "b");
  EXPECT_EQ(registry.segment_names()[1].second[1], "c");

  EXPECT_FLOAT_EQ(*registry.providers()[0].second().values[0]->GetTensorData<float>(), 1.0f);
  ASSERT_EQ(registry.providers()[1].second().values.size(), 2u);
  EXPECT_FLOAT_EQ(*registry.providers()[1].second().values[0]->GetTensorData<float>(), 2.0f);
}

TEST(HistoryManagerTest, ObservationAndActionLengthsIndependent)
{
  HistoryManager hm;
  hm.set_lengths(2, 1);

  std::vector<float> obs_raw = {1.0f};
  std::vector<float> obs2_raw = {2.0f};
  std::vector<float> act_raw = {9.0f};
  auto obs_ort = make_ort_values(obs_raw);
  auto obs2_ort = make_ort_values(obs2_raw);
  auto act_ort = make_ort_values(act_raw);

  hm.push_observation(obs_ort);
  hm.push_observation(obs2_ort);
  hm.push_action(act_ort);

  ASSERT_EQ(hm.observations().size(), 2u);
  ASSERT_EQ(hm.actions().size(), 1u);
  EXPECT_FLOAT_EQ(*hm.actions()[0][0]->GetTensorData<float>(), 9.0f);
}

TEST(HistoryManagerTest, SetLengthsToZeroClearsBuffers)
{
  HistoryManager hm;
  hm.set_lengths(5, 5);

  std::vector<float> obs_raw = {1.0f};
  std::vector<float> act_raw = {2.0f};
  auto obs_ort = make_ort_values(obs_raw);
  auto act_ort = make_ort_values(act_raw);

  hm.push_observation(obs_ort);
  hm.push_action(act_ort);
  hm.set_lengths(0, 0);
  EXPECT_TRUE(hm.observations().empty());
  EXPECT_TRUE(hm.actions().empty());
}

/// Scalar data-type tests
// Each test verifies that a 1-D tensor of the given element type round-trips
// through HistoryManager without corruption or type coercion.

TEST(DataTypeTest, DoubleTensor)
{
  const std::vector<int64_t> shape = {3};
  const std::vector<double> data = {1.1, 2.2, 3.3};
  auto tensor = make_typed_tensor<double>(data, shape);

  EXPECT_EQ(
    tensor->GetTensorTypeAndShapeInfo().GetElementType(),
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE);

  const double * ptr = tensor->GetTensorData<double>();
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_DOUBLE_EQ(ptr[i], data[i]);
  }

  HistoryManager hm;
  hm.set_lengths(1, 0);
  hm.push_observation({tensor});
  ASSERT_EQ(hm.observations().size(), 1u);
  EXPECT_DOUBLE_EQ(*hm.observations()[0][0]->GetTensorData<double>(), data[0]);
}

TEST(DataTypeTest, Int32Tensor)
{
  const std::vector<int64_t> shape = {4};
  const std::vector<int32_t> data = {-100, 0, 100, std::numeric_limits<int32_t>::max()};
  auto tensor = make_typed_tensor<int32_t>(data, shape);

  EXPECT_EQ(
    tensor->GetTensorTypeAndShapeInfo().GetElementType(),
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);

  const int32_t * ptr = tensor->GetTensorData<int32_t>();
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(ptr[i], data[i]);
  }

  HistoryManager hm;
  hm.set_lengths(1, 0);
  hm.push_observation({tensor});
  EXPECT_EQ(*hm.observations()[0][0]->GetTensorData<int32_t>(), data[0]);
}

TEST(DataTypeTest, Int64Tensor)
{
  // int64 is the canonical type for discrete action indices and token IDs.
  const std::vector<int64_t> shape = {3};
  const std::vector<int64_t> data = {0LL, 42LL, std::numeric_limits<int64_t>::max()};
  auto tensor = make_typed_tensor<int64_t>(data, shape);

  EXPECT_EQ(
    tensor->GetTensorTypeAndShapeInfo().GetElementType(),
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);

  const int64_t * ptr = tensor->GetTensorData<int64_t>();
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(ptr[i], data[i]);
  }

  HistoryManager hm;
  hm.set_lengths(1, 0);
  hm.push_observation({tensor});
  EXPECT_EQ(*hm.observations()[0][0]->GetTensorData<int64_t>(), data[0]);
}

TEST(DataTypeTest, Int16Tensor)
{
  const std::vector<int64_t> shape = {2};
  const std::vector<int16_t> data = {-32768, 32767};
  auto tensor = make_typed_tensor<int16_t>(data, shape);

  EXPECT_EQ(
    tensor->GetTensorTypeAndShapeInfo().GetElementType(),
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16);

  const int16_t * ptr = tensor->GetTensorData<int16_t>();
  EXPECT_EQ(ptr[0], data[0]);
  EXPECT_EQ(ptr[1], data[1]);
}

TEST(DataTypeTest, Int8Tensor)
{
  const std::vector<int64_t> shape = {2};
  const std::vector<int8_t> data = {-128, 127};
  auto tensor = make_typed_tensor<int8_t>(data, shape);

  EXPECT_EQ(
    tensor->GetTensorTypeAndShapeInfo().GetElementType(),
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8);

  const int8_t * ptr = tensor->GetTensorData<int8_t>();
  EXPECT_EQ(ptr[0], data[0]);
  EXPECT_EQ(ptr[1], data[1]);
}

TEST(DataTypeTest, UInt8Tensor)
{
  // uint8 is the standard element type for raw pixel data.
  const std::vector<int64_t> shape = {3};
  const std::vector<uint8_t> data = {0u, 128u, 255u};
  auto tensor = make_typed_tensor<uint8_t>(data, shape);

  EXPECT_EQ(
    tensor->GetTensorTypeAndShapeInfo().GetElementType(),
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

  const uint8_t * ptr = tensor->GetTensorData<uint8_t>();
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(ptr[i], data[i]);
  }
}

TEST(DataTypeTest, UInt16Tensor)
{
  // uint16 is typical for 16-bit depth images (e.g. RealSense, Kinect).
  const std::vector<int64_t> shape = {2};
  const std::vector<uint16_t> data = {0u, 65535u};
  auto tensor = make_typed_tensor<uint16_t>(data, shape);

  EXPECT_EQ(
    tensor->GetTensorTypeAndShapeInfo().GetElementType(),
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16);

  const uint16_t * ptr = tensor->GetTensorData<uint16_t>();
  EXPECT_EQ(ptr[0], data[0]);
  EXPECT_EQ(ptr[1], data[1]);
}

TEST(DataTypeTest, UInt32Tensor)
{
  const std::vector<int64_t> shape = {2};
  const std::vector<uint32_t> data = {0u, std::numeric_limits<uint32_t>::max()};
  auto tensor = make_typed_tensor<uint32_t>(data, shape);

  EXPECT_EQ(
    tensor->GetTensorTypeAndShapeInfo().GetElementType(),
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32);

  const uint32_t * ptr = tensor->GetTensorData<uint32_t>();
  EXPECT_EQ(ptr[0], data[0]);
  EXPECT_EQ(ptr[1], data[1]);
}

TEST(DataTypeTest, UInt64Tensor)
{
  const std::vector<int64_t> shape = {2};
  const std::vector<uint64_t> data = {0ull, std::numeric_limits<uint64_t>::max()};
  auto tensor = make_typed_tensor<uint64_t>(data, shape);

  EXPECT_EQ(
    tensor->GetTensorTypeAndShapeInfo().GetElementType(),
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64);

  const uint64_t * ptr = tensor->GetTensorData<uint64_t>();
  EXPECT_EQ(ptr[0], data[0]);
  EXPECT_EQ(ptr[1], data[1]);
}

// Image / multi-dimensional tensor tests
// Physical-AI pipelines frequently feed whole images directly into policies.
// These tests confirm that HWC / CHW / NCHW tensors survive the
// ObservationProviderRegistry to HistoryManager round-trip intact.

TEST(ImageDataTest, RGBImageUInt8_HWC)
{
  // Simulate a small 4×4 RGB image in HWC layout (H=4, W=4, C=3).
  const int64_t H = 4, W = 4, C = 3;
  const std::vector<int64_t> shape = {H, W, C};
  const int64_t n_elems = shape_size(shape);

  std::vector<uint8_t> pixels(n_elems);
  std::iota(pixels.begin(), pixels.end(), uint8_t{0});
  auto tensor = make_typed_tensor<uint8_t>(pixels, shape);

  EXPECT_EQ(
    tensor->GetTensorTypeAndShapeInfo().GetElementType(),
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
  EXPECT_EQ(tensor->GetTensorTypeAndShapeInfo().GetShape(), shape);
  EXPECT_EQ(tensor->GetTensorTypeAndShapeInfo().GetElementCount(), static_cast<size_t>(n_elems));

  const uint8_t * ptr = tensor->GetTensorData<uint8_t>();
  for (int64_t i = 0; i < n_elems; ++i) {
    EXPECT_EQ(ptr[i], static_cast<uint8_t>(i));
  }

  HistoryManager hm;
  hm.set_lengths(1, 0);
  hm.push_observation({tensor});
  ASSERT_EQ(hm.observations().size(), 1u);
  EXPECT_EQ(
    hm.observations()[0][0]->GetTensorTypeAndShapeInfo().GetShape(), shape);
}

TEST(ImageDataTest, DepthImageFloat_HW)
{
  // Simulate a 4×4 float depth map (metric metres, HW layout).
  const int64_t H = 4, W = 4;
  const std::vector<int64_t> shape = {H, W};
  const int64_t n_elems = shape_size(shape);

  std::vector<float> depth(n_elems);
  for (int64_t i = 0; i < n_elems; ++i) {
    depth[i] = static_cast<float>(i) * 0.1f;
  }
  auto tensor = make_typed_tensor<float>(depth, shape);

  EXPECT_EQ(
    tensor->GetTensorTypeAndShapeInfo().GetElementType(),
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  EXPECT_EQ(tensor->GetTensorTypeAndShapeInfo().GetShape(), shape);

  const float * ptr = tensor->GetTensorData<float>();
  EXPECT_FLOAT_EQ(ptr[0], 0.0f);
  EXPECT_FLOAT_EQ(ptr[n_elems - 1], static_cast<float>(n_elems - 1) * 0.1f);

  HistoryManager hm;
  hm.set_lengths(1, 0);
  hm.push_observation({tensor});
  EXPECT_EQ(
    hm.observations()[0][0]->GetTensorTypeAndShapeInfo().GetShape(), shape);
}

TEST(ImageDataTest, DepthImageUInt16_HW)
{
  // Simulate a 4×4 uint16 depth image as returned by a raw depth sensor.
  const int64_t H = 4, W = 4;
  const std::vector<int64_t> shape = {H, W};
  const int64_t n_elems = shape_size(shape);

  std::vector<uint16_t> depth(n_elems);
  std::iota(depth.begin(), depth.end(), uint16_t{1000});
  auto tensor = make_typed_tensor<uint16_t>(depth, shape);

  EXPECT_EQ(
    tensor->GetTensorTypeAndShapeInfo().GetElementType(),
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16);
  EXPECT_EQ(tensor->GetTensorTypeAndShapeInfo().GetShape(), shape);

  const uint16_t * ptr = tensor->GetTensorData<uint16_t>();
  EXPECT_EQ(ptr[0], uint16_t{1000});
  EXPECT_EQ(ptr[n_elems - 1], static_cast<uint16_t>(1000 + n_elems - 1));
}

TEST(ImageDataTest, NormalizedNCHWFloat)
{
  // Typical normalised image tensor fed to vision models: [N=1, C=3, H=4, W=4].
  const int64_t N = 1, C = 3, H = 4, W = 4;
  const std::vector<int64_t> shape = {N, C, H, W};
  const int64_t n_elems = shape_size(shape);

  std::vector<float> img(n_elems);
  for (int64_t i = 0; i < n_elems; ++i) {
    img[i] = static_cast<float>(i) / static_cast<float>(n_elems - 1);  // [0, 1]
  }
  auto tensor = make_typed_tensor<float>(img, shape);

  EXPECT_EQ(
    tensor->GetTensorTypeAndShapeInfo().GetElementType(),
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  EXPECT_EQ(tensor->GetTensorTypeAndShapeInfo().GetShape(), shape);
  EXPECT_FLOAT_EQ(tensor->GetTensorData<float>()[0], 0.0f);
  EXPECT_FLOAT_EQ(tensor->GetTensorData<float>()[n_elems - 1], 1.0f);

  HistoryManager hm;
  hm.set_lengths(2, 0);
  hm.push_observation({tensor});
  hm.push_observation({tensor});  // duplicate frame simulates two consecutive steps
  ASSERT_EQ(hm.observations().size(), 2u);
  EXPECT_EQ(
    hm.observations()[0][0]->GetTensorTypeAndShapeInfo().GetShape(), shape);
}

// Mixed-type provider tests

TEST(HistoryManagerTest, MultiDimTensorRoundTrip)
{
  // Verify that 2-D joint-space matrices (e.g. [joints=7, features=3]) survive.
  const std::vector<int64_t> shape = {7, 3};
  const int64_t n_elems = shape_size(shape);
  std::vector<float> data(n_elems);
  std::iota(data.begin(), data.end(), 0.5f);

  auto tensor = make_typed_tensor<float>(data, shape);

  HistoryManager hm;
  hm.set_lengths(3, 0);
  hm.push_observation({tensor});

  ASSERT_EQ(hm.observations().size(), 1u);
  const auto & retrieved = hm.observations()[0][0];
  EXPECT_EQ(retrieved->GetTensorTypeAndShapeInfo().GetShape(), shape);
  const float * ptr = retrieved->GetTensorData<float>();
  for (int64_t i = 0; i < n_elems; ++i) {
    EXPECT_FLOAT_EQ(ptr[i], data[i]);
  }
}

TEST(RegistryTest, MixedTypeProviders)
{
  // Registry holding providers that return different ONNX element types —
  // typical in physical AI where joint states (float) and camera frames
  // (uint8) come from different sources.

  const std::vector<int64_t> float_shape = {6};    // 6-DOF joint positions
  const std::vector<int64_t> image_shape = {4, 4, 3};  // small HWC RGB patch

  std::vector<float> joint_data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
  std::vector<uint8_t> pixel_data(shape_size(image_shape));
  std::iota(pixel_data.begin(), pixel_data.end(), uint8_t{10});

  auto joint_tensor = make_typed_tensor<float>(joint_data, float_shape);
  auto image_tensor = make_typed_tensor<uint8_t>(pixel_data, image_shape);

  std::vector<std::shared_ptr<Ort::Value>> joint_vec = {joint_tensor};
  std::vector<std::shared_ptr<Ort::Value>> image_vec = {image_tensor};

  rclcpp::Time t(0, 0);
  ObservationData joint_obs{joint_vec, t};
  ObservationData image_obs{image_vec, t};

  ObservationProviderRegistry registry;
  registry.register_provider(
    "joint_states", {"q1", "q2", "q3", "q4", "q5", "q6"},
    [&joint_obs]() -> const ObservationData & {return joint_obs;});
  registry.register_provider(
    "rgb_camera", {"image"},
    [&image_obs]() -> const ObservationData & {return image_obs;});

  ASSERT_EQ(registry.providers().size(), 2u);

  const auto & joint_result = registry.providers()[0].second();
  ASSERT_EQ(joint_result.values.size(), 1u);
  EXPECT_EQ(
    joint_result.values[0]->GetTensorTypeAndShapeInfo().GetElementType(),
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  EXPECT_EQ(joint_result.values[0]->GetTensorTypeAndShapeInfo().GetShape(), float_shape);
  EXPECT_FLOAT_EQ(joint_result.values[0]->GetTensorData<float>()[0], 0.1f);

  const auto & image_result = registry.providers()[1].second();
  ASSERT_EQ(image_result.values.size(), 1u);
  EXPECT_EQ(
    image_result.values[0]->GetTensorTypeAndShapeInfo().GetElementType(),
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
  EXPECT_EQ(image_result.values[0]->GetTensorTypeAndShapeInfo().GetShape(), image_shape);
  EXPECT_EQ(image_result.values[0]->GetTensorData<uint8_t>()[0], uint8_t{10});
}

TEST(HistoryManagerTest, MixedTypeTensorsInSingleObservation)
{
  // A single observation snapshot may bundle different tensor types
  // (e.g. float joint state + uint8 image + int64 discrete token).
  const std::vector<int64_t> fshape = {6};
  const std::vector<int64_t> ishape = {2, 2, 3};
  const std::vector<int64_t> tshape = {1};

  std::vector<float> jdata = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<uint8_t> pdata(shape_size(ishape), uint8_t{255});
  std::vector<int64_t> tdata = {7LL};

  auto joint_tensor = make_typed_tensor<float>(jdata, fshape);
  auto image_tensor = make_typed_tensor<uint8_t>(pdata, ishape);
  auto token_tensor = make_typed_tensor<int64_t>(tdata, tshape);

  std::vector<std::shared_ptr<Ort::Value>> obs = {joint_tensor, image_tensor, token_tensor};

  HistoryManager hm;
  hm.set_lengths(2, 0);
  hm.push_observation(obs);

  ASSERT_EQ(hm.observations().size(), 1u);
  const auto & snap = hm.observations()[0];
  ASSERT_EQ(snap.size(), 3u);

  EXPECT_EQ(
    snap[0]->GetTensorTypeAndShapeInfo().GetElementType(),
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  EXPECT_FLOAT_EQ(snap[0]->GetTensorData<float>()[0], 1.0f);

  EXPECT_EQ(
    snap[1]->GetTensorTypeAndShapeInfo().GetElementType(),
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
  EXPECT_EQ(snap[1]->GetTensorData<uint8_t>()[0], uint8_t{255});

  EXPECT_EQ(
    snap[2]->GetTensorTypeAndShapeInfo().GetElementType(),
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  EXPECT_EQ(snap[2]->GetTensorData<int64_t>()[0], int64_t{7});
}

}  // namespace ros2_policy_execution_core
