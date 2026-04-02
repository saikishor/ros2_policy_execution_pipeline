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

#ifndef ROS2_POLICY_EXECUTION_CORE__VALUE_TYPES_HPP_
#define ROS2_POLICY_EXECUTION_CORE__VALUE_TYPES_HPP_

#include <cstddef>
#include <cstdint>
#include <span>  // NOLINT(build/include_order)

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace ros2_policy_execution_core
{

/**
 * @brief Supported element types for dense tensors in the policy pipeline.
 */
enum class DataType
{
  kUnknown = 0,
  kFloat32,
  kFloat64,
  kInt32,
  kInt64,
  kUInt8,
  kBool
};

/**
 * @brief Return the size in bytes of a datatype, or zero when unknown.
 *
 * @param[in] data_type Datatype to inspect.
 * @return Size in bytes, or zero for `kUnknown`.
 */
constexpr std::size_t data_type_size(const DataType data_type)
{
  switch (data_type) {
    case DataType::kFloat32:
      return sizeof(float);
    case DataType::kFloat64:
      return sizeof(double);
    case DataType::kInt32:
      return sizeof(int32_t);
    case DataType::kInt64:
      return sizeof(int64_t);
    case DataType::kUInt8:
      return sizeof(uint8_t);
    case DataType::kBool:
      return sizeof(bool);
    case DataType::kUnknown:
      return 0;
  }
  return 0;
}

template<typename T>
struct DataTypeFromCpp
{
  static constexpr DataType value = DataType::kUnknown;
};

template<>
struct DataTypeFromCpp<float>
{
  static constexpr DataType value = DataType::kFloat32;
};

template<>
struct DataTypeFromCpp<double>
{
  static constexpr DataType value = DataType::kFloat64;
};

template<>
struct DataTypeFromCpp<int32_t>
{
  static constexpr DataType value = DataType::kInt32;
};

template<>
struct DataTypeFromCpp<int64_t>
{
  static constexpr DataType value = DataType::kInt64;
};

template<>
struct DataTypeFromCpp<uint8_t>
{
  static constexpr DataType value = DataType::kUInt8;
};

template<>
struct DataTypeFromCpp<bool>
{
  static constexpr DataType value = DataType::kBool;
};

/**
 * @brief Map a C++ element type to the corresponding policy-execution datatype.
 *
 * A `switch` cannot be used here because the input is a C++ type, not a runtime
 * value. Template specializations keep the mapping explicit and easy to extend.
 *
 * @tparam T C++ element type.
 * @return Matching datatype, or `kUnknown` if unsupported.
 */
template<typename T>
constexpr DataType data_type_from_cpp()
{
  return DataTypeFromCpp<std::remove_cv_t<T>>::value;
}

/**
 * @brief Device type metadata for future backend-specific execution.
 */
enum class DeviceType
{
  kCpu = 0,
  kCuda,
  kCustom
};

/**
 * @brief Lightweight device description attached to a tensor.
 */
struct Device
{
  DeviceType type = DeviceType::kCpu;
  int device_id = 0;
};

/**
 * @brief Shared or borrowed storage backing one or more tensors.
 *
 * `SharedBuffer` keeps lifetime separate from tensor metadata. This allows a
 * preprocessor to allocate once, and later code to create multiple tensor views
 * over the same memory without copying payload bytes.
 *
 * The key idea is:
 * - `data_` points at the bytes to read or write.
 * - `owner_` optionally keeps those bytes alive.
 *
 * When the bytes are owned somewhere else, such as a ROS message or
 * `std::vector`, the buffer can "borrow" the pointer and store a shared owner.
 * The tensor then does not own the bytes itself, but it can safely use them for
 * as long as the shared owner is kept alive.
 */
class SharedBuffer
{
public:
  /**
   * @brief Construct an empty buffer.
   */
  SharedBuffer() = default;

  /**
   * @brief Construct a buffer view over mutable storage.
   *
   * The constructor stores its own `std::shared_ptr`. In the usual call pattern the
   * caller passes `owner`, not `std::move(owner)`, so the caller keeps its own shared
   * pointer and the buffer stores another handle to the same payload.
   *
   * Example:
   *
   * \code{.cpp}
   * auto values = std::make_shared<std::vector<float>>(
   *   std::initializer_list<float>{1.0f, 2.0f, 3.0f});
   *
   * SharedBuffer buffer(values->data(), values->size() * sizeof(float), values);
   *
   * // `values` is still valid here. Both `values` and `buffer.owner()` now
   * // share ownership of the same vector storage.
   * \endcode
   *
   * If the caller instead passes `std::move(values)`, then the caller gives its
   * shared-pointer handle to the buffer and `values` becomes empty.
   *
   * @param[in] data Start of the storage.
   * @param[in] bytes Size of the storage in bytes.
   * @param[in] owner Optional lifetime anchor for shared or borrowed storage.
   */
  SharedBuffer(
    void * data, std::size_t bytes,
    std::shared_ptr<void> owner = {})
  : owner_(std::move(owner)),
    data_(static_cast<const std::byte *>(data)),
    bytes_(bytes),
    is_mutable_(true)
  {}

  /**
   * @brief Construct a buffer view over read-only storage.
   *
   * This overload is for genuinely const input memory. Mutable accessors will
   * reject such buffers at runtime.
   *
   * @param[in] data Start of the storage.
   * @param[in] bytes Size of the storage in bytes.
   * @param[in] owner Optional lifetime anchor for shared or borrowed storage.
   */
  SharedBuffer(
    const void * data, std::size_t bytes,
    std::shared_ptr<const void> owner = {})
  : owner_(std::move(owner)),
    data_(static_cast<const std::byte *>(data)),
    bytes_(bytes),
    is_mutable_(false)
  {}

  /**
   * @brief Create a buffer that borrows a mutable typed memory range.
   *
   * Note: "Borrow" means the buffer does not allocate or copy the bytes. It only
   * stores the pointer and, optionally, a shared owner that guarantees the
   * pointed-to memory remains valid.
   *
   * @tparam T Element type.
   * @param[in] data Start of the typed storage.
   * @param[in] count Number of elements.
   * @param[in] owner Optional lifetime anchor.
   * @return Buffer view covering the provided range.
   */
  template<typename T, typename = std::enable_if_t<!std::is_const_v<T>>>
  static SharedBuffer borrow(
    T * data, const std::size_t count,
    std::shared_ptr<void> owner = {})
  {
    return SharedBuffer(data, count * sizeof(T), std::move(owner));
  }

  /**
   * @brief Create a buffer that borrows a read-only typed memory range.
   *
   * @tparam T Element type.
   * @param[in] data Start of the typed storage.
   * @param[in] count Number of elements.
   * @param[in] owner Optional lifetime anchor.
   * @return Buffer view covering the provided range.
   */
  template<typename T>
  static SharedBuffer borrow(
    const T * data, const std::size_t count,
    std::shared_ptr<const void> owner = {})
  {
    return SharedBuffer(data, count * sizeof(T), std::move(owner));
  }

  /**
   * @brief Create a buffer view over a shared `std::vector` without copying.
   *
   * Note: The returned buffer points directly at `values->data()` and stores another
   * `std::shared_ptr` to the same vector, so the vector storage remains alive
   * even if the original caller later releases its copy of the shared pointer.
   *
   * @tparam T Vector element type.
   * @param[in] values Shared vector that owns the storage.
   * @return Buffer referencing the vector payload.
   */
  template<typename T>
  static SharedBuffer share_vector(const std::shared_ptr<std::vector<T>> & values)
  {
    if (!values) {
      throw std::invalid_argument("Cannot build a SharedBuffer from a null vector.");
    }
    if (values->empty()) {
      return SharedBuffer(
        static_cast<void *>(nullptr), 0, std::static_pointer_cast<void>(values));
    }
    return SharedBuffer(
      values->data(),
      values->size() * sizeof(T),
      std::static_pointer_cast<void>(values));
  }

  /**
   * @brief Create a buffer view over mutable shared storage described by another owner.
   *
   * Note: This is useful when the bytes live inside a larger object, such as a ROS
   * message. The buffer aliases the payload pointer but keeps the whole message
   * alive through the shared owner.
   *
   * @tparam OwnerT Shared owner type.
   * @param[in] owner Shared object that keeps the storage alive.
   * @param[in] data Start of the storage.
   * @param[in] bytes Size of the storage in bytes.
   * @return Buffer view covering the provided range.
   */
  template<typename OwnerT>
  static SharedBuffer alias(
    const std::shared_ptr<OwnerT> & owner,
    void * data, const std::size_t bytes)
  {
    if (!owner) {
      throw std::invalid_argument("Cannot alias a null owner.");
    }
    return SharedBuffer(
      data,
      bytes,
      std::static_pointer_cast<void>(owner));
  }

  /**
   * @brief Create a buffer view over read-only shared storage described by another owner.
   *
   * @tparam OwnerT Shared owner type.
   * @param[in] owner Shared object that keeps the storage alive.
   * @param[in] data Start of the storage.
   * @param[in] bytes Size of the storage in bytes.
   * @return Buffer view covering the provided range.
   */
  template<typename OwnerT>
  static SharedBuffer alias(
    const std::shared_ptr<OwnerT> & owner,
    const void * data, const std::size_t bytes)
  {
    if (!owner) {
      throw std::invalid_argument("Cannot alias a null owner.");
    }
    return SharedBuffer(data, bytes, owner);
  }

  /**
   * @brief Return the start of the storage as a read-only pointer.
   *
   * @return Pointer to the start of the storage.
   */
  [[nodiscard]] const void * data() const
  {
    return data_;
  }

  /**
   * @brief Return the start of the storage as a mutable pointer.
   *
   * This is mainly used by inference backends that need a non-const API for
   * inputs, such as ONNX Runtime's tensor creation helpers.
   *
   * @return Mutable pointer to the start of the storage.
   */
  [[nodiscard]] void * mutable_data()
  {
    if (!is_mutable_) {
      throw std::runtime_error("Cannot request mutable data from an immutable SharedBuffer.");
    }
    return const_cast<std::byte *>(data_);
  }

  /**
   * @brief Return whether the underlying storage may be accessed mutably.
   *
   * @return `true` when the buffer wraps mutable storage.
   */
  [[nodiscard]] bool is_mutable() const
  {
    return is_mutable_;
  }

  /**
   * @brief Return the size of the storage in bytes.
   *
   * @return Byte size of the storage.
   */
  [[nodiscard]] std::size_t bytes() const
  {
    return bytes_;
  }

  /**
   * @brief Return whether this buffer references valid storage.
   *
   * @return `true` when the buffer has storage or represents an empty view.
   */
  [[nodiscard]] bool is_valid() const
  {
    return data_ != nullptr || bytes_ == 0;
  }

  /**
   * @brief Return the lifetime anchor associated with the storage.
   *
   * @return Shared owner that keeps the storage alive.
   */
  [[nodiscard]] const std::shared_ptr<const void> & owner() const
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
 * @brief Dense tensor metadata plus a view into shared storage.
 *
 * v1 intentionally focuses on dense tensors. `strides` and `byte_offset` are
 * included so the type can represent lightweight subviews, but most helpers
 * assume contiguous storage.
 */
class Tensor
{
public:
  /**
   * @brief Construct an empty tensor.
   */
  Tensor() = default;

  /**
   * @brief Construct a dense tensor view over existing storage.
   *
   * @param[in] data_type Tensor element type.
   * @param[in] shape Tensor shape.
   * @param[in] buffer Storage backing the tensor.
   * @param[in] device Device metadata.
   * @param[in] strides Optional strides in elements.
   * @param[in] byte_offset Byte offset into the shared buffer.
   */
  Tensor(
    DataType data_type,
    std::vector<int64_t> shape,
    SharedBuffer buffer,
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
   * @brief Create a tensor view over a shared `std::vector` without copying.
   *
   * @tparam T Vector element type.
   * @param[in] values Shared vector that owns the payload.
   * @param[in] shape Tensor shape.
   * @param[in] device Device metadata.
   * @return Tensor view over the vector payload.
   */
  template<typename T>
  static Tensor share_vector(
    const std::shared_ptr<std::vector<T>> & values,
    std::vector<int64_t> shape,
    const Device device = {})
  {
    return Tensor(
      data_type_from_cpp<T>(),
      std::move(shape),
      SharedBuffer::share_vector(values),
      device);
  }

  /**
   * @brief Return the tensor datatype.
   *
   * @return Tensor datatype.
   */
  [[nodiscard]] DataType data_type() const
  {
    return data_type_;
  }

  /**
   * @brief Return the tensor shape.
   *
   * @return Tensor shape.
   */
  [[nodiscard]] const std::vector<int64_t> & shape() const
  {
    return shape_;
  }

  /**
   * @brief Return the tensor strides in elements.
   *
   * An empty vector means contiguous layout.
   *
   * @return Tensor strides.
   */
  [[nodiscard]] const std::vector<int64_t> & strides() const
  {
    return strides_;
  }

  /**
   * @brief Return the backing buffer.
   *
   * @return Buffer describing the underlying storage.
   */
  [[nodiscard]] const SharedBuffer & buffer() const
  {
    return buffer_;
  }

  /**
   * @brief Return the device metadata.
   *
   * @return Device metadata.
   */
  [[nodiscard]] const Device & device() const
  {
    return device_;
  }

  /**
   * @brief Return the byte offset into the backing buffer.
   *
   * @return Byte offset from the start of the buffer.
   */
  [[nodiscard]] std::size_t byte_offset() const
  {
    return byte_offset_;
  }

  /**
   * @brief Return whether the tensor is contiguous.
   *
   * @return `true` when the tensor has no custom strides.
   */
  [[nodiscard]] bool is_contiguous() const
  {
    return strides_.empty();
  }

  /**
   * @brief Return the tensor rank.
   *
   * @return Number of dimensions.
   */
  [[nodiscard]] std::size_t rank() const
  {
    return shape_.size();
  }

  /**
   * @brief Return the number of elements described by the shape.
   *
   * Scalars use zero dimensions and therefore contain exactly one element.
   *
   * @return Number of tensor elements.
   */
  [[nodiscard]] std::size_t num_elements() const
  {
    std::size_t count = 1;
    for (const auto dimension : shape_) {
      count *= static_cast<std::size_t>(dimension);
    }
    return count;
  }

  /**
   * @brief Return the logical size of the tensor payload in bytes.
   *
   * @return Logical tensor payload size in bytes.
   */
  [[nodiscard]] std::size_t bytes() const
  {
    return num_elements() * data_type_size(data_type_);
  }

  /**
   * @brief Return the number of bytes still available after the offset.
   *
   * @return Remaining bytes in the backing buffer.
   */
  [[nodiscard]] std::size_t available_bytes() const
  {
    if (byte_offset_ > buffer_.bytes()) {
      return 0;
    }
    return buffer_.bytes() - byte_offset_;
  }

  /**
   * @brief Return the read-only payload pointer.
   *
   * @return Pointer to the tensor payload.
   */
  [[nodiscard]] const void * raw_data() const
  {
    if (buffer_.data() == nullptr) {
      return nullptr;
    }
    return static_cast<const std::byte *>(buffer_.data()) + byte_offset_;
  }

  /**
   * @brief Return the mutable payload pointer.
   *
   * @return Mutable pointer to the tensor payload.
   */
  [[nodiscard]] void * mutable_raw_data()
  {
    if (buffer_.data() == nullptr) {
      return nullptr;
    }
    return static_cast<std::byte *>(buffer_.mutable_data()) + byte_offset_;
  }

  /**
   * @brief Return a contiguous typed span over the payload.
   *
   * @tparam T Requested element type.
   * @return Mutable span over the tensor payload.
   */
  template<typename T>
  [[nodiscard]] std::span<T> span()
  {
    validate_span_type<T>();
    return std::span<T>(static_cast<T *>(mutable_raw_data()), num_elements());
  }

  /**
   * @brief Return a contiguous typed span over the payload.
   *
   * @tparam T Requested element type.
   * @return Read-only span over the tensor payload.
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
    if (data_type_ == DataType::kUnknown) {
      throw std::invalid_argument("Tensor datatype must be known.");
    }
    for (const auto dimension : shape_) {
      if (dimension < 0) {
        throw std::invalid_argument("Tensor dimensions must be non-negative.");
      }
    }
    if (byte_offset_ > buffer_.bytes()) {
      throw std::invalid_argument("Tensor byte offset exceeds the buffer size.");
    }
    if (is_contiguous() && bytes() > available_bytes()) {
      throw std::invalid_argument("Tensor payload does not fit in the backing buffer.");
    }
  }

  template<typename T>
  void validate_span_type() const
  {
    if (!is_contiguous()) {
      throw std::runtime_error("Typed spans require contiguous tensors.");
    }
    if (data_type_ != data_type_from_cpp<T>()) {
      throw std::runtime_error("Requested span element type does not match the tensor datatype.");
    }
  }

  DataType data_type_ = DataType::kUnknown;
  std::vector<int64_t> shape_ = {};
  std::vector<int64_t> strides_ = {};
  SharedBuffer buffer_ = {};
  Device device_ = {};
  std::size_t byte_offset_ = 0;
};

/**
 * @brief Transport value shared between preprocessors, inference backends, and
 * postprocessors.
 *
 * v1 only contains dense tensors, but the wrapper leaves room for future value
 * kinds without forcing a breaking API change now.
 */
class Value
{
public:
  /**
   * @brief Kinds supported by the value wrapper.
   */
  enum class Kind
  {
    kEmpty = 0,
    kTensor
  };

  /**
   * @brief Construct an empty value.
   */
  Value() = default;

  /**
   * @brief Construct a value from a tensor.
   *
   * @param[in] tensor Tensor payload.
   */
  explicit Value(Tensor tensor)
  : payload_(std::move(tensor))
  {}

  /**
   * @brief Return the contained value kind.
   *
   * @return Current value kind.
   */
  [[nodiscard]] Kind kind() const
  {
    if (std::holds_alternative<Tensor>(payload_)) {
      return Kind::kTensor;
    }
    return Kind::kEmpty;
  }

  /**
   * @brief Return whether the value contains a tensor.
   *
   * @return `true` when the value contains a tensor.
   */
  [[nodiscard]] bool is_tensor() const
  {
    return std::holds_alternative<Tensor>(payload_);
  }

  /**
   * @brief Return the contained tensor.
   *
   * @return Contained tensor.
   */
  [[nodiscard]] const Tensor & as_tensor() const
  {
    return std::get<Tensor>(payload_);
  }

  /**
   * @brief Return the contained tensor.
   *
   * @return Contained tensor.
   */
  [[nodiscard]] Tensor & as_tensor()
  {
    return std::get<Tensor>(payload_);
  }

private:
  std::variant<std::monostate, Tensor> payload_ = {};
};

/**
 * @brief Named value entry used for ordered model inputs and outputs.
 */
struct NamedValue
{
  std::string name;
  Value value;
};

/**
 * @brief Ordered container for named model inputs and outputs.
 */
using ValueSet = std::vector<NamedValue>;

/**
 * @brief Find a named value in a `ValueSet`.
 *
 * @param[in] values Ordered set of named values.
 * @param[in] name Name to look up.
 * @return Pointer to the matching value, or `nullptr`.
 */
inline const Value * find_value(const ValueSet & values, const std::string & name)
{
  for (const auto & entry : values) {
    if (entry.name == name) {
      return &entry.value;
    }
  }
  return nullptr;
}

/**
 * @brief Find a named value in a `ValueSet`.
 *
 * @param[in] values Ordered set of named values.
 * @param[in] name Name to look up.
 * @return Pointer to the matching value, or `nullptr`.
 */
inline Value * find_value(ValueSet & values, const std::string & name)
{
  for (auto & entry : values) {
    if (entry.name == name) {
      return &entry.value;
    }
  }
  return nullptr;
}

}  // namespace ros2_policy_execution_core

#endif  // ROS2_POLICY_EXECUTION_CORE__VALUE_TYPES_HPP_
