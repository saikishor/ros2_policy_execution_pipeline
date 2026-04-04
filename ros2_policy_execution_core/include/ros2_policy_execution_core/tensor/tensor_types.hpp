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

#ifndef ROS2_POLICY_EXECUTION_CORE__TENSOR__TENSOR_TYPES_HPP_
#define ROS2_POLICY_EXECUTION_CORE__TENSOR__TENSOR_TYPES_HPP_

/**
 * @file tensor_types.hpp
 * @brief `ByteBufferView` (byte range + optional lifetime anchor) and dense `Tensor` views.
 *
 * \sa tensor/data_type.hpp for \ref DataType and `data_type_v` without pulling in buffer types.
 */

#include <cstddef>
#include <span>  // NOLINT(build/include_order)

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "ros2_policy_execution_core/tensor/data_type.hpp"
#include "ros2_policy_execution_core/tensor/tensor_device.hpp"

namespace ros2_policy_execution_core
{

/**
 * @brief Non-owning view of a byte range (`data`, `bytes`) with an optional `std::shared_ptr` anchor.
 *
 * Does not allocate the payload: `owner()` may be empty when the caller guarantees lifetime, or
 * non-empty to keep storage alive (for example shared `std::vector` storage, a ROS message body,
 * or an ONNX Runtime wrapper). One allocation may back several `Tensor` views; mutability is fixed
 * at construction (mutable vs read-only overloads).
 */
class ByteBufferView
{
public:
  /// @brief Constructs an empty view (no storage).
  ByteBufferView() = default;

  /**
   * @brief Constructs a view over mutable storage.
   *
   * When \p owner is non-empty, its lifetime extends that of \p data. Passing \p owner
   * by lvalue copies the std::shared_ptr; std::move(owner) transfers the handle into
   * this object and leaves the source empty.
   *
   * \code{.cpp}
   * auto values = std::make_shared<std::vector<float>>(
   *   std::initializer_list<float>{1.0f, 2.0f, 3.0f});
   *
   * ByteBufferView buffer(values->data(), values->size() * sizeof(float), values);
   *
   * // `values` remains valid; `values` and `buffer.owner()` reference the same control block.
   * \endcode
   *
   * @param[in] data Pointer to the first byte of the storage.
   * @param[in] bytes Size of the storage in bytes.
   * @param[in] owner Optional shared owner whose lifetime covers \p data.
   */
  ByteBufferView(
    void * data, std::size_t bytes,
    std::shared_ptr<void> owner = {})
  : owner_(std::move(owner)),
    data_(static_cast<const std::byte *>(data)),
    bytes_(bytes),
    is_mutable_(true)
  {}

  /**
   * @brief Constructs a view over read-only storage.
   *
   * Buffers constructed through this overload are immutable; mutable_data() fails at runtime.
   *
   * @param[in] data Pointer to the first byte of the storage.
   * @param[in] bytes Size of the storage in bytes.
   * @param[in] owner Optional shared owner whose lifetime covers \p data.
   */
  ByteBufferView(
    const void * data, std::size_t bytes,
    std::shared_ptr<const void> owner = {})
  : owner_(std::move(owner)),
    data_(static_cast<const std::byte *>(data)),
    bytes_(bytes),
    is_mutable_(false)
  {}

  /**
   * @brief Non-owning view over a mutable typed range; does not allocate or copy elements.
   * @tparam T Element type.
   * @param[in] data Start of the typed storage.
   * @param[in] count Number of elements.
   * @param[in] owner Optional shared owner keeping \p data valid.
   */
  template<typename T>
  requires(!std::is_const_v<T>)
  [[nodiscard]] static ByteBufferView borrow(
    T * data, const std::size_t count,
    std::shared_ptr<void> owner = {})
  {
    return ByteBufferView(data, count * sizeof(T), std::move(owner));
  }

  /**
   * @brief Non-owning view over a const typed range; does not allocate or copy elements.
   * @tparam T Element type.
   * @param[in] data Start of the typed storage.
   * @param[in] count Number of elements.
   * @param[in] owner Optional shared owner keeping \p data valid.
   */
  template<typename T>
  [[nodiscard]] static ByteBufferView borrow(
    const T * data, const std::size_t count,
    std::shared_ptr<const void> owner = {})
  {
    return ByteBufferView(data, count * sizeof(T), std::move(owner));
  }

  /**
   * @brief View over std::vector storage held by a shared pointer; aliases vector::data() and shares ownership.
   * @tparam T Vector element type.
   * @param[in] values Non-null shared vector owning the bytes.
   */
  template<typename T>
  [[nodiscard]] static ByteBufferView share_vector(const std::shared_ptr<std::vector<T>> & values)
  {
    if (!values) {
      throw std::invalid_argument("Cannot build a ByteBufferView from a null vector.");
    }
    if (values->empty()) {
      return ByteBufferView(
        static_cast<void *>(nullptr), 0, std::static_pointer_cast<void>(values));
    }
    return ByteBufferView(
      values->data(),
      values->size() * sizeof(T),
      std::static_pointer_cast<void>(values));
  }

  /**
   * @brief Mutable payload inside a larger shared object (for example, a ROS message body).
   * @tparam OwnerT Shared owner type.
   * @param[in] owner Non-null shared object whose lifetime covers \p data.
   * @param[in] data Start of the aliased bytes.
   * @param[in] bytes Size of the aliased region in bytes.
   */
  template<typename OwnerT>
  [[nodiscard]] static ByteBufferView alias(
    const std::shared_ptr<OwnerT> & owner,
    void * data, const std::size_t bytes)
  {
    if (!owner) {
      throw std::invalid_argument("Cannot alias a null owner.");
    }
    return ByteBufferView(
      data,
      bytes,
      std::static_pointer_cast<void>(owner));
  }

  /**
   * @brief Const payload inside a larger shared object (for example, a ROS message body).
   * @tparam OwnerT Shared owner type.
   * @param[in] owner Non-null shared object whose lifetime covers \p data.
   * @param[in] data Start of the aliased bytes.
   * @param[in] bytes Size of the aliased region in bytes.
   */
  template<typename OwnerT>
  [[nodiscard]] static ByteBufferView alias(
    const std::shared_ptr<OwnerT> & owner,
    const void * data, const std::size_t bytes)
  {
    if (!owner) {
      throw std::invalid_argument("Cannot alias a null owner.");
    }
    return ByteBufferView(data, bytes, owner);
  }

  /// @brief Read-only start of the byte range.
  [[nodiscard]] const void * data() const noexcept
  {
    return data_;
  }

  /**
   * @brief Mutable pointer to the start of the byte range.
   *
   * Raises std::runtime_error if the buffer was constructed from read-only storage.
   * Inference backends may use this when a non-const pointer is required for wrapped storage.
   */
  [[nodiscard]] void * mutable_data()
  {
    if (!is_mutable_) {
      throw std::runtime_error("Cannot request mutable data from an immutable ByteBufferView.");
    }
    return const_cast<std::byte *>(data_);
  }

  /// @brief Whether mutable_data() may be invoked without raising std::runtime_error.
  [[nodiscard]] bool is_mutable() const noexcept
  {
    return is_mutable_;
  }

  /// @brief Length of the byte range.
  [[nodiscard]] std::size_t bytes() const noexcept
  {
    return bytes_;
  }

  /// @brief Whether data() is non-null or the byte length is zero.
  [[nodiscard]] bool is_valid() const noexcept
  {
    return data_ != nullptr || bytes_ == 0;
  }

  /// @brief Shared owner keeping storage alive, if any.
  [[nodiscard]] const std::shared_ptr<const void> & owner() const noexcept
  {
    return owner_;
  }

private:
  std::shared_ptr<const void> owner_ = {};
  const std::byte * data_ = nullptr;
  std::size_t bytes_ = 0;
  bool is_mutable_ = false;
};

/**
 * @brief Dense tensor view: shape, optional stride placeholder, datatype, device, and `ByteBufferView` storage view.
 *
 * The pipeline contract is contiguous row-major storage only: `strides()` must be empty at construction.
 * Non-empty strides are rejected in validate(). Use `byte_offset` to address a subrange inside the buffer.
 * A future strided view type may be added without changing this dense invariant.
 */
class Tensor
{
public:
  /// @brief Default-constructed tensor; must be assigned before use as a non-empty payload.
  Tensor() = default;

  /**
   * @brief Constructs a tensor view over existing storage and invokes validate().
   * @param[in] data_type Element datatype.
   * @param[in] shape Per-dimension sizes (non-negative).
   * @param[in] buffer Backing bytes (must cover payload for contiguous tensors).
   * @param[in] device Device metadata.
   * @param[in] strides Must be empty (dense row-major). Non-empty values are rejected.
   * @param[in] byte_offset Byte offset from the start of \p buffer to the tensor payload.
   */
  Tensor(
    DataType data_type,
    std::vector<int64_t> shape,
    ByteBufferView buffer,
    Device device = {},
    std::vector<int64_t> strides = {},
    std::size_t byte_offset = 0)
  : data_type_(data_type),
    shape_(std::move(shape)),
    strides_(std::move(strides)),
    buffer_(std::move(buffer)),
    device_(device),
    byte_offset_(byte_offset)
  {
    validate();
  }

  /**
   * @brief Factory equivalent to Tensor with ByteBufferView::share_vector; element type is \p T.
   * @tparam T std::vector element type.
   * @param[in] values Non-null shared vector owning the payload.
   * @param[in] shape Tensor dimensions.
   * @param[in] device Device metadata.
   */
  template<typename T>
  [[nodiscard]] static Tensor share_vector(
    const std::shared_ptr<std::vector<T>> & values,
    std::vector<int64_t> shape,
    const Device device = {})
  {
    return Tensor(
      data_type_v<T>,
      std::move(shape),
      ByteBufferView::share_vector(values),
      device);
  }

  /// @brief Element datatype of the tensor.
  [[nodiscard]] DataType data_type() const noexcept
  {
    return data_type_;
  }

  /// @brief Per-dimension extents.
  [[nodiscard]] const std::vector<int64_t> & shape() const noexcept
  {
    return shape_;
  }

  /// @brief Empty after validate(); placeholder field for a future strided tensor type.
  [[nodiscard]] const std::vector<int64_t> & strides() const noexcept
  {
    return strides_;
  }

  /// @brief Underlying byte storage and lifetime anchor.
  [[nodiscard]] const ByteBufferView & buffer() const noexcept
  {
    return buffer_;
  }

  /// @brief Device placement metadata.
  [[nodiscard]] const Device & device() const noexcept
  {
    return device_;
  }

  /// @brief Payload start as a byte offset into buffer().
  [[nodiscard]] std::size_t byte_offset() const noexcept
  {
    return byte_offset_;
  }

  /// @brief True when strides() is empty; always true for tensors that passed validate().
  [[nodiscard]] bool is_contiguous() const noexcept
  {
    return strides_.empty();
  }

  /// @brief Number of dimensions (shape().size()).
  [[nodiscard]] std::size_t rank() const noexcept
  {
    return shape_.size();
  }

  /// @brief Product of shape dimensions (scalar rank 0 yields one element).
  [[nodiscard]] std::size_t num_elements() const noexcept
  {
    std::size_t count = 1;
    for (const auto dimension : shape_) {
      count *= static_cast<std::size_t>(dimension);
    }
    return count;
  }

  /// @brief Logical payload size: num_elements() * element size in bytes.
  [[nodiscard]] std::size_t bytes() const noexcept
  {
    return num_elements() * data_type_size(data_type_);
  }

  /// @brief Bytes from byte_offset() to the end of buffer() (clamped if offset past end).
  [[nodiscard]] std::size_t available_bytes() const noexcept
  {
    if (byte_offset_ > buffer_.bytes()) {
      return 0;
    }
    return buffer_.bytes() - byte_offset_;
  }

  /// @brief Read-only tensor payload (buffer data + byte_offset); may be nullptr.
  [[nodiscard]] const void * raw_data() const noexcept
  {
    if (buffer_.data() == nullptr) {
      return nullptr;
    }
    return static_cast<const std::byte *>(buffer_.data()) + byte_offset_;
  }

  /// @brief Mutable tensor payload; uses buffer().mutable_data().
  [[nodiscard]] void * mutable_raw_data()
  {
    if (buffer_.data() == nullptr) {
      return nullptr;
    }
    return static_cast<std::byte *>(buffer_.mutable_data()) + byte_offset_;
  }

  /**
   * @brief Contiguous mutable span; type must match data_type().
   * @tparam T Element type.
   */
  template<typename T>
  [[nodiscard]] std::span<T> span()
  {
    validate_span_type<T>();
    return std::span<T>(static_cast<T *>(mutable_raw_data()), num_elements());
  }

  /**
   * @brief Contiguous const span; type must match data_type().
   * @tparam T Element type.
   */
  template<typename T>
  [[nodiscard]] std::span<const T> span() const
  {
    validate_span_type<T>();
    return std::span<const T>(static_cast<const T *>(raw_data()), num_elements());
  }

private:
  void validate() const
  {
    if (!buffer_.is_valid()) {
      throw std::invalid_argument("Tensor storage is not valid.");
    }
    if (data_type_ == DataType::Unknown) {
      throw std::invalid_argument("Tensor datatype must be known.");
    }
    for (const auto dimension : shape_) {
      if (dimension < 0) {
        throw std::invalid_argument("Tensor dimensions must be non-negative.");
      }
    }
    if (!strides_.empty()) {
      throw std::invalid_argument(
        "Tensor strides are not supported; use dense row-major storage (empty strides).");
    }
    if (byte_offset_ > buffer_.bytes()) {
      throw std::invalid_argument("Tensor byte offset exceeds the buffer size.");
    }
    if (bytes() > available_bytes()) {
      throw std::invalid_argument("Tensor payload does not fit in the backing buffer.");
    }
  }

  template<typename T>
  void validate_span_type() const
  {
    if (data_type_ != data_type_v<T>) {
      throw std::runtime_error("Requested span element type does not match the tensor datatype.");
    }
  }

  DataType data_type_ = DataType::Unknown;
  std::vector<int64_t> shape_ = {};
  std::vector<int64_t> strides_ = {};
  ByteBufferView buffer_ = {};
  Device device_ = {};
  std::size_t byte_offset_ = 0;
};

}  // namespace ros2_policy_execution_core

#endif  // ROS2_POLICY_EXECUTION_CORE__TENSOR__TENSOR_TYPES_HPP_
