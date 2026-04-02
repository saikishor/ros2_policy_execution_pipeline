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
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "ros2_policy_execution_core/value_types.hpp"

namespace ros2_policy_execution_core
{

namespace
{

int64_t channels_from_encoding(const std::string & encoding)
{
  if (encoding == "mono8") {
    return 1;
  }
  if (encoding == "rgb8" || encoding == "bgr8") {
    return 3;
  }
  if (encoding == "rgba8" || encoding == "bgra8") {
    return 4;
  }
  throw std::runtime_error("Unsupported image encoding for the example node: " + encoding);
}

}  // namespace

/**
 * @brief Example ROS 2 node that turns heterogeneous messages into a `ValueSet`.
 *
 * The image tensor wraps the ROS message payload without copying it.
 * The pose tensor stores seven `double` values in a compact shared array.
 */
class ExamplePreprocessorNode : public rclcpp::Node
{
public:
  /**
   * @brief Construct the example node and create the subscriptions.
   */
  ExamplePreprocessorNode()
  : Node("example_preprocessor_value_node")
  {
    // Subscribe to an image topic whose payload can later be wrapped directly
    // as a `uint8` tensor without copying the image bytes.
    image_subscription_ = create_subscription<sensor_msgs::msg::Image>(
      "image",
      rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::Image::SharedPtr message) {
        latest_image_ = std::move(message);
        publish_example_summary();
      });

    // Subscribe to a pose topic and later condense it into a compact 7-element
    // numeric tensor.
    pose_subscription_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "tcp_pose",
      rclcpp::SystemDefaultsQoS(),
      [this](geometry_msgs::msg::PoseStamped::SharedPtr message) {
        latest_pose_ = std::move(message);
        publish_example_summary();
      });
  }

private:
  /**
   * @brief Build a `ValueSet` from the latest ROS messages.
   *
   * @return Ordered named values ready for a future inference backend.
   */
  [[nodiscard]] ValueSet build_values() const
  {
    ValueSet values;

    // Add the image tensor if an image message has already been received.
    if (latest_image_) {
      values.push_back({"image", Value(make_image_tensor(latest_image_))});
    }
    // Add the pose tensor if a pose message has already been received.
    if (latest_pose_) {
      values.push_back({"tcp_pose", Value(make_pose_tensor(latest_pose_))});
    }

    return values;
  }

  /**
   * @brief Wrap an image message as a tensor without copying payload bytes.
   *
   * @param[in] image Image message to wrap.
   * @return Tensor view over the ROS message payload.
   */
  [[nodiscard]] Tensor make_image_tensor(
    const sensor_msgs::msg::Image::SharedPtr & image) const
  {
    // Reuse the byte payload stored in the ROS message directly. The shared
    // message pointer becomes the buffer owner so the bytes stay valid.
    return Tensor(
      DataType::kUInt8,
      {
        static_cast<int64_t>(image->height),
        static_cast<int64_t>(image->width),
        channels_from_encoding(image->encoding)
      },
      SharedBuffer::alias(image, image->data.data(), image->data.size()));
  }

  /**
   * @brief Convert a pose message into a compact numeric tensor.
   *
   * @param[in] pose Pose message to convert.
   * @return Tensor containing position and quaternion values.
   */
  [[nodiscard]] Tensor make_pose_tensor(
    const geometry_msgs::msg::PoseStamped::SharedPtr & pose) const
  {
    // Pack the heterogeneous pose fields into one compact numeric block that is
    // easier for an inference backend to consume.
    auto pose_values = std::make_shared<std::array<double, 7>>(std::array<double, 7>{
        pose->pose.position.x,
        pose->pose.position.y,
        pose->pose.position.z,
        pose->pose.orientation.x,
        pose->pose.orientation.y,
        pose->pose.orientation.z,
        pose->pose.orientation.w
    });

    return Tensor(
      DataType::kFloat64,
      {7},
      // The tensor then aliases that contiguous array without another copy.
      SharedBuffer::alias(pose_values, pose_values->data(), sizeof(double) * pose_values->size()));
  }

  /**
   * @brief Log a short summary each time enough data is available to build a value set.
   */
  void publish_example_summary()
  {
    if (!latest_image_ && !latest_pose_) {
      return;
    }

    // This mirrors what a real preprocessor would hand to the next stage:
    // an ordered set of named tensors with backend-agnostic metadata.
    const auto values = build_values();
    RCLCPP_INFO(
      get_logger(),
      "Built ValueSet with %zu entries.%s%s",
      values.size(),
      find_value(values, "image") != nullptr ? " image" : "",
      find_value(values, "tcp_pose") != nullptr ? " tcp_pose" : "");
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_subscription_;
  sensor_msgs::msg::Image::SharedPtr latest_image_;
  geometry_msgs::msg::PoseStamped::SharedPtr latest_pose_;
};

}  // namespace ros2_policy_execution_core

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ros2_policy_execution_core::ExamplePreprocessorNode>());
  rclcpp::shutdown();
  return 0;
}
