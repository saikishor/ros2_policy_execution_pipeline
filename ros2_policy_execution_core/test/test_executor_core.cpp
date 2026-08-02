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

#include <memory>
#include <vector>

#include "ros2_policy_execution_core/onnxruntime_types.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"

#include "ros2_policy_execution_core/executor_core.hpp"

namespace ros2_policy_execution_core
{

namespace
{
/// Creates one 1-D single-element Ort::Value tensor per float, using ORT's allocator.
/// The returned shared_ptrs own their memory — no external backing data required.
std::vector<OrtValueSharedPtr> make_ort_values(const std::vector<float> & data)
{
  static Ort::AllocatorWithDefaultOptions allocator;
  std::vector<int64_t> shape = {1};
  std::vector<OrtValueSharedPtr> values;
  values.reserve(data.size());
  for (float v : data) {
    auto tensor = std::make_shared<Ort::Value>(
      Ort::Value::CreateTensor<float>(allocator, shape.data(), 1));
    *tensor->GetTensorMutableData<float>() = v;
    values.push_back(std::move(tensor));
  }
  return values;
}
}  // namespace

/**
 * @brief A simple implementation of ExecutorCore for testing.
 */
class TestableExecutorCore : public ExecutorCore
{
public:
  void configure(const rclcpp::Node::SharedPtr & /*node*/) override
  {
    configured_ = true;
  }

  bool execute(const std::vector<OrtValueSharedPtr> & commands) override
  {
    last_commands_ = commands;
    return succeed_;
  }

  bool is_configured() const {return configured_;}
  const std::vector<OrtValueSharedPtr> & last_commands() const {return last_commands_;}
  void set_should_succeed(bool succeed) {succeed_ = succeed;}

private:
  bool configured_ = false;
  bool succeed_ = true;
  std::vector<OrtValueSharedPtr> last_commands_;
};

class ExecutorCoreTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    rclcpp::init(0, nullptr);
    executor_ = std::make_unique<TestableExecutorCore>();
  }

  void TearDown() override
  {
    executor_.reset();
    rclcpp::shutdown();
  }

  std::unique_ptr<TestableExecutorCore> executor_;
};

TEST_F(ExecutorCoreTest, ConfigureCallsDerivedImplementation)
{
  auto node = std::make_shared<rclcpp::Node>("test_node");
  EXPECT_FALSE(executor_->is_configured());
  executor_->configure(node);
  EXPECT_TRUE(executor_->is_configured());
}

TEST_F(ExecutorCoreTest, ExecuteReceivesExactCommandsPassedIn)
{
  std::vector<float> command_raw = {1.5f, -2.5f, 3.0f};
  auto commands = make_ort_values(command_raw);

  EXPECT_TRUE(executor_->execute(commands));

  const auto & received = executor_->last_commands();
  ASSERT_EQ(received.size(), command_raw.size());
  for (size_t i = 0; i < command_raw.size(); ++i) {
    EXPECT_FLOAT_EQ(*received[i]->GetTensorData<float>(), command_raw[i]);
  }
}

TEST_F(ExecutorCoreTest, ExecutePropagatesFailure)
{
  executor_->set_should_succeed(false);
  std::vector<float> command_raw = {0.0f};
  auto commands = make_ort_values(command_raw);

  EXPECT_FALSE(executor_->execute(commands));
}

TEST_F(ExecutorCoreTest, UsableThroughBasePointer)
{
  std::unique_ptr<ExecutorCore> base = std::make_unique<TestableExecutorCore>();
  auto node = std::make_shared<rclcpp::Node>("test_node");
  base->configure(node);

  std::vector<float> command_raw = {42.0f};
  auto commands = make_ort_values(command_raw);
  EXPECT_TRUE(base->execute(commands));
}

TEST_F(ExecutorCoreTest, DestructionThroughBasePointerRunsDerivedDestructor)
{
  static bool destroyed = false;
  destroyed = false;

  class DestructorTrackingExecutor : public ExecutorCore
  {
public:
    void configure(const rclcpp::Node::SharedPtr & /*node*/) override {}
    bool execute(const std::vector<OrtValueSharedPtr> & /*commands*/) override {return true;}
    ~DestructorTrackingExecutor() override {destroyed = true;}
  };

  {
    std::unique_ptr<ExecutorCore> base = std::make_unique<DestructorTrackingExecutor>();
  }

  EXPECT_TRUE(destroyed);
}

}  // namespace ros2_policy_execution_core
