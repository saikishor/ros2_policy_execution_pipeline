// Copyright 2026 PAL Robotics S.L.
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

/// \author Sai Kishor Kothakota

#include <exception>
#include <memory>

#include "rclcpp/rclcpp.hpp"

#include "ros2_policy_execution_pipeline/policy_execution_pipeline.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>("policy_execution_pipeline");

  try {
    auto pipeline =
      std::make_shared<ros2_policy_execution_pipeline::PolicyExecutionPipeline>(node);
    pipeline->init();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(
      node->get_logger(), "Failed to start policy execution pipeline: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
