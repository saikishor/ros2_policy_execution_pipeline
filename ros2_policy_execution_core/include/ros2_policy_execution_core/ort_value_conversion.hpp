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

#ifndef ROS2_POLICY_EXECUTION_CORE__ORT_VALUE_CONVERSION_HPP_
#define ROS2_POLICY_EXECUTION_CORE__ORT_VALUE_CONVERSION_HPP_

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ros2_policy_execution_core/value_types.hpp"

namespace ros2_policy_execution_core
{

/**
 * @brief Small wrapper that keeps borrowed tensor memory alive while exposing
 * it as an `Ort::Value`.
 *
 * ONNX Runtime can wrap caller-provided CPU buffers without copying them, but
 * the caller must keep that storage alive for as long as the `Ort::Value` is
 * used. This wrapper makes that lifetime explicit.
 */
struct OrtValueReference
{
  Ort::Value value = Ort::Value{nullptr};
  std::shared_ptr<const void> owner = {};
};

/**
 * @brief Convert a policy-execution datatype to the matching ONNX Runtime type.
 *
 * @param[in] data_type Datatype to convert.
 * @return Matching ONNX Runtime tensor element type.
 */
inline ONNXTensorElementDataType to_onnx_data_type(const DataType data_type)
{
  switch (data_type) {
    case DataType::kFloat32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case DataType::kFloat64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    case DataType::kInt32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case DataType::kInt64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case DataType::kUInt8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case DataType::kBool:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    case DataType::kUnknown:
      break;
  }
  throw std::invalid_argument("Unsupported policy tensor datatype for ONNX Runtime.");
}

/**
 * @brief Convert an ONNX Runtime element type to the matching policy datatype.
 *
 * @param[in] data_type ONNX Runtime element type.
 * @return Matching policy datatype.
 */
inline DataType from_onnx_data_type(const ONNXTensorElementDataType data_type)
{
  switch (data_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return DataType::kFloat32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return DataType::kFloat64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return DataType::kInt32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return DataType::kInt64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return DataType::kUInt8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return DataType::kBool;
    default:
      break;
  }
  throw std::invalid_argument("Unsupported ONNX Runtime tensor datatype.");
}

/**
 * @brief Wrap a dense CPU tensor as an `Ort::Value` without copying payload
 * bytes.
 *
 * @param[in] tensor Mutable tensor to expose to ONNX Runtime.
 * @return Wrapper containing the `Ort::Value` and the storage lifetime anchor.
 */
inline OrtValueReference make_ort_value_reference(Tensor & tensor)
{
  if (!tensor.is_contiguous()) {
    throw std::invalid_argument("ONNX Runtime conversion requires a contiguous tensor.");
  }
  if (tensor.device().type != DeviceType::kCpu) {
    throw std::invalid_argument("v1 ONNX Runtime conversion only supports CPU tensors.");
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
 * @brief Create a non-owning tensor view over an `Ort::Value`.
 *
 * The caller must keep the `Ort::Value` alive while using the returned tensor.
 *
 * @param[in] value ONNX Runtime tensor value.
 * @return Tensor view over the existing payload.
 */
inline Tensor borrow_tensor_from_ort_value(const Ort::Value & value)
{
  if (!value.IsTensor()) {
    throw std::invalid_argument("Only dense ONNX Runtime tensors are supported.");
  }

  const auto shape_info = value.GetTensorTypeAndShapeInfo();
  const auto memory_info = value.GetTensorMemoryInfo();
  const auto device_type = memory_info.GetDeviceType();
  if (device_type != OrtMemoryInfoDeviceType_CPU) {
    throw std::invalid_argument("v1 only supports ONNX Runtime tensors in CPU memory.");
  }

  Device device;
  device.type = DeviceType::kCpu;
  device.device_id = memory_info.GetDeviceId();

  return Tensor(
    from_onnx_data_type(shape_info.GetElementType()),
    shape_info.GetShape(),
    SharedBuffer(value.GetTensorRawData(), value.GetTensorSizeInBytes()),
    device);
}

/**
 * @brief Create a tensor that keeps the moved `Ort::Value` alive internally.
 *
 * This is useful when a postprocessor wants to consume ONNX Runtime outputs via
 * the generic tensor abstraction without copying payload bytes.
 *
 * @param[in] value ONNX Runtime tensor value to wrap.
 * @return Tensor view whose backing storage remains valid while the tensor lives.
 */
inline Tensor tensor_from_ort_value(Ort::Value value)
{
  if (!value.IsTensor()) {
    throw std::invalid_argument("Only dense ONNX Runtime tensors are supported.");
  }

  auto owner = std::make_shared<Ort::Value>(std::move(value));
  const auto shape_info = owner->GetTensorTypeAndShapeInfo();
  const auto memory_info = owner->GetTensorMemoryInfo();
  const auto device_type = memory_info.GetDeviceType();
  if (device_type != OrtMemoryInfoDeviceType_CPU) {
    throw std::invalid_argument("v1 only supports ONNX Runtime tensors in CPU memory.");
  }

  Device device;
  device.type = DeviceType::kCpu;
  device.device_id = memory_info.GetDeviceId();

  return Tensor(
    from_onnx_data_type(shape_info.GetElementType()),
    shape_info.GetShape(),
    SharedBuffer(owner->GetTensorRawData(), owner->GetTensorSizeInBytes(), owner),
    device);
}

/**
 * @brief Create a tensor that keeps both an `Ort::Value` and its original
 * borrowed owner alive internally.
 *
 * This overload is the safest round-trip path for a tensor that was first
 * wrapped by `make_ort_value_reference()`.
 *
 * @param[in] reference ONNX Runtime tensor value plus the lifetime anchor for
 * the borrowed source storage.
 * @return Tensor view whose backing storage remains valid while the tensor lives.
 */
inline Tensor tensor_from_ort_value(OrtValueReference reference)
{
  if (!reference.value.IsTensor()) {
    throw std::invalid_argument("Only dense ONNX Runtime tensors are supported.");
  }

  auto owner = std::make_shared<OrtValueReference>(std::move(reference));
  const auto shape_info = owner->value.GetTensorTypeAndShapeInfo();
  const auto memory_info = owner->value.GetTensorMemoryInfo();
  const auto device_type = memory_info.GetDeviceType();
  if (device_type != OrtMemoryInfoDeviceType_CPU) {
    throw std::invalid_argument("v1 only supports ONNX Runtime tensors in CPU memory.");
  }

  Device device;
  device.type = DeviceType::kCpu;
  device.device_id = memory_info.GetDeviceId();

  return Tensor(
    from_onnx_data_type(shape_info.GetElementType()),
    shape_info.GetShape(),
    SharedBuffer(
      owner->value.GetTensorRawData(),
      owner->value.GetTensorSizeInBytes(),
      owner),
    device);
}

}  // namespace ros2_policy_execution_core

#endif  // ROS2_POLICY_EXECUTION_CORE__ORT_VALUE_CONVERSION_HPP_
