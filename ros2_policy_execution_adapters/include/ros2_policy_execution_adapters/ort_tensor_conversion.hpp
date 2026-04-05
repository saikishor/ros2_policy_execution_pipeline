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

#ifndef ROS2_POLICY_EXECUTION_ADAPTERS__ORT_TENSOR_CONVERSION_HPP_
#define ROS2_POLICY_EXECUTION_ADAPTERS__ORT_TENSOR_CONVERSION_HPP_

/**
 * @file ort_tensor_conversion.hpp
 * @brief `Tensor` ↔ ONNX Runtime `Ort::Value` conversion (CPU dense tensors).
 */

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ros2_policy_execution_core/tensor/tensor_types.hpp"

namespace ros2_policy_execution_adapters
{

/**
 * @brief Pairs an `Ort::Value` with the shared owner keeping its borrowed memory alive.
 */
struct OrtValueReference
{
  Ort::Value value{nullptr};
  std::shared_ptr<const void> owner = {};
};

/**
 * @brief Convert a pipeline DataType to the corresponding ONNX Runtime element type.
 * @param[in] data_type Pipeline tensor element type.
 */
inline ONNXTensorElementDataType to_onnx_data_type(
  const ros2_policy_execution_core::DataType data_type)
{
  using ros2_policy_execution_core::DataType;
  switch (data_type) {
    case DataType::Float16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case DataType::BFloat16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    case DataType::Float32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case DataType::Float64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    case DataType::Int8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case DataType::Int16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    case DataType::Int32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case DataType::Int64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case DataType::UInt8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case DataType::UInt16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    case DataType::UInt32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    case DataType::UInt64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    case DataType::Bool:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    default:
      break;
  }
  throw std::invalid_argument("Unsupported policy tensor datatype for ONNX Runtime.");
}

/**
 * @brief Convert an ONNX Runtime element type to the corresponding pipeline DataType.
 * @param[in] data_type ONNX Runtime element type.
 */
inline ros2_policy_execution_core::DataType from_onnx_data_type(
  const ONNXTensorElementDataType data_type)
{
  using ros2_policy_execution_core::DataType;
  switch (data_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return DataType::Float16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return DataType::BFloat16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return DataType::Float32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return DataType::Float64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return DataType::Int8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return DataType::Int16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return DataType::Int32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return DataType::Int64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return DataType::UInt8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return DataType::UInt16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return DataType::UInt32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return DataType::UInt64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return DataType::Bool;
    default:
      break;
  }
  throw std::invalid_argument("Unsupported ONNX Runtime tensor datatype.");
}

/**
 * @brief Wrap a CPU tensor as an `Ort::Value` without copying; requires mutable storage.
 * @param[in] tensor Mutable CPU tensor; caller retains ownership of the underlying buffer.
 */
inline OrtValueReference make_ort_value_reference(ros2_policy_execution_core::Tensor & tensor)
{
  using ros2_policy_execution_core::DataType;
  using ros2_policy_execution_core::Tensor;
  if (tensor.device().type != ros2_policy_execution_core::DeviceType::Cpu) {
    throw std::invalid_argument("ONNX Runtime conversion only supports CPU tensors.");
  }
  if (!tensor.buffer().is_mutable()) {
    throw std::invalid_argument(
            "ONNX Runtime zero-copy wrapping requires mutable tensor storage.");
  }
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto value = Ort::Value::CreateTensor(
    memory_info,
    tensor.mutable_raw_data(),
    tensor.bytes(),
    tensor.shape().data(),
    tensor.shape().size(),
    to_onnx_data_type(tensor.data_type()));
  return OrtValueReference{std::move(value), tensor.buffer().owner()};
}

/**
 * @brief Non-owning read-only Tensor view over an `Ort::Value`.
 * @param[in] value CPU tensor value; caller must keep alive while the returned Tensor is in use.
 */
inline ros2_policy_execution_core::Tensor borrow_tensor_from_ort_value(const Ort::Value & value)
{
  using ros2_policy_execution_core::Device;
  using ros2_policy_execution_core::DeviceType;
  using ros2_policy_execution_core::ByteBufferView;
  using ros2_policy_execution_core::Tensor;
  if (!value.IsTensor()) {
    throw std::invalid_argument("Only dense ONNX Runtime tensors are supported.");
  }
  const Ort::ConstMemoryInfo mem_info = value.GetTensorMemoryInfo();
  if (mem_info.GetDeviceType() != OrtMemoryInfoDeviceType_CPU) {
    throw std::invalid_argument("Only ONNX Runtime tensors in CPU memory are supported.");
  }
  auto shape_info = value.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> shape = shape_info.GetShape();
  const std::size_t bytes =
    shape_info.GetElementCount() * ros2_policy_execution_core::data_type_size(
    from_onnx_data_type(shape_info.GetElementType()));
  return Tensor(
    from_onnx_data_type(shape_info.GetElementType()),
    std::move(shape),
    ByteBufferView(value.GetTensorRawData(), bytes),
    Device{DeviceType::Cpu, 0});
}

/**
 * @brief Non-owning mutable Tensor view over an `Ort::Value`.
 * @param[in] value Mutable CPU tensor value; caller must keep alive while the returned Tensor is in use.
 */
inline ros2_policy_execution_core::Tensor borrow_tensor_from_ort_value(Ort::Value & value)
{
  using ros2_policy_execution_core::Device;
  using ros2_policy_execution_core::DeviceType;
  using ros2_policy_execution_core::ByteBufferView;
  using ros2_policy_execution_core::Tensor;
  if (!value.IsTensor()) {
    throw std::invalid_argument("Only dense ONNX Runtime tensors are supported.");
  }
  const Ort::ConstMemoryInfo mem_info = value.GetTensorMemoryInfo();
  if (mem_info.GetDeviceType() != OrtMemoryInfoDeviceType_CPU) {
    throw std::invalid_argument("Only ONNX Runtime tensors in CPU memory are supported.");
  }
  auto shape_info = value.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> shape = shape_info.GetShape();
  const std::size_t bytes =
    shape_info.GetElementCount() * ros2_policy_execution_core::data_type_size(
    from_onnx_data_type(shape_info.GetElementType()));
  return Tensor(
    from_onnx_data_type(shape_info.GetElementType()),
    std::move(shape),
    ByteBufferView(value.GetTensorMutableRawData(), bytes),
    Device{DeviceType::Cpu, 0});
}

/**
 * @brief Owning Tensor that keeps the moved `Ort::Value` alive internally.
 * @param[in] value ORT tensor value; moved into shared storage owned by the returned Tensor.
 */
inline ros2_policy_execution_core::Tensor tensor_from_ort_value(Ort::Value value)
{
  using ros2_policy_execution_core::Device;
  using ros2_policy_execution_core::DeviceType;
  using ros2_policy_execution_core::ByteBufferView;
  using ros2_policy_execution_core::Tensor;
  if (!value.IsTensor()) {
    throw std::invalid_argument("Only dense ONNX Runtime tensors are supported.");
  }
  auto owner = std::make_shared<Ort::Value>(std::move(value));
  const Ort::ConstMemoryInfo mem_info = owner->GetTensorMemoryInfo();
  if (mem_info.GetDeviceType() != OrtMemoryInfoDeviceType_CPU) {
    throw std::invalid_argument("Only ONNX Runtime tensors in CPU memory are supported.");
  }
  auto shape_info = owner->GetTensorTypeAndShapeInfo();
  std::vector<int64_t> shape = shape_info.GetShape();
  const std::size_t bytes =
    shape_info.GetElementCount() * ros2_policy_execution_core::data_type_size(
    from_onnx_data_type(shape_info.GetElementType()));
  return Tensor(
    from_onnx_data_type(shape_info.GetElementType()),
    std::move(shape),
    ByteBufferView::alias(owner, owner->GetTensorMutableRawData(), bytes),
    Device{DeviceType::Cpu, 0});
}

/**
 * @brief Owning Tensor that keeps both the `Ort::Value` and the buffer owner alive.
 * @param[in] reference ORT tensor value plus the lifetime anchor for the payload bytes.
 */
inline ros2_policy_execution_core::Tensor tensor_from_ort_value(OrtValueReference reference)
{
  using ros2_policy_execution_core::Device;
  using ros2_policy_execution_core::DeviceType;
  using ros2_policy_execution_core::ByteBufferView;
  using ros2_policy_execution_core::Tensor;
  if (!reference.value.IsTensor()) {
    throw std::invalid_argument("Only dense ONNX Runtime tensors are supported.");
  }
  auto owner = std::make_shared<OrtValueReference>(std::move(reference));
  const Ort::ConstMemoryInfo mem_info = owner->value.GetTensorMemoryInfo();
  if (mem_info.GetDeviceType() != OrtMemoryInfoDeviceType_CPU) {
    throw std::invalid_argument("Only ONNX Runtime tensors in CPU memory are supported.");
  }
  auto shape_info = owner->value.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> shape = shape_info.GetShape();
  const std::size_t bytes =
    shape_info.GetElementCount() * ros2_policy_execution_core::data_type_size(
    from_onnx_data_type(shape_info.GetElementType()));
  return Tensor(
    from_onnx_data_type(shape_info.GetElementType()),
    std::move(shape),
    ByteBufferView::alias(owner, owner->value.GetTensorMutableRawData(), bytes),
    Device{DeviceType::Cpu, 0});
}

}  // namespace ros2_policy_execution_adapters

#endif  // ROS2_POLICY_EXECUTION_ADAPTERS__ORT_TENSOR_CONVERSION_HPP_
