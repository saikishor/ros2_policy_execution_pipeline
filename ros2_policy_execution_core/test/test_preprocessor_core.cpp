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

#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"

#include "ros2_policy_execution_core/preprocessor_core.hpp"
#include "ros2_policy_execution_core/tensor/named_value_list.hpp"

namespace ros2_policy_execution_core
{

/**
 * @brief A simple implementation of PreprocessorCore for testing.
 */
class TestablePreprocessorCore : public PreprocessorCore
{
public:
  void configure(const rclcpp::Node::SharedPtr & /*node*/) override
  {
    configured_ = true;
  }

  bool is_configured() const {return configured_;}

private:
  bool configured_ = false;
};

class PreprocessorCoreTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    rclcpp::init(0, nullptr);
    preprocessor_ = std::make_unique<TestablePreprocessorCore>();
  }

  void TearDown() override
  {
    preprocessor_.reset();
    rclcpp::shutdown();
  }

  std::unique_ptr<TestablePreprocessorCore> preprocessor_;
};

TEST_F(PreprocessorCoreTest, ConfigureCallsDerivedImplementation)
{
  auto node = std::make_shared<rclcpp::Node>("test_node");
  EXPECT_FALSE(preprocessor_->is_configured());
  preprocessor_->configure(node);
  EXPECT_TRUE(preprocessor_->is_configured());
}

TEST_F(PreprocessorCoreTest, SetConfigSetsHistoryLengths)
{
  PreprocessorCoreConfig config;
  config.observation_history_length = 5;
  config.action_history_length = 3;

  preprocessor_->set_config(config);

  // Verify by checking that history is stored up to the configured length
  std::vector<float> obs = {1.0f, 2.0f};
  for (size_t i = 0; i < 10; ++i) {
    preprocessor_->set_previous_observations(obs);
  }
  // Should only keep 5 observations
  EXPECT_EQ(preprocessor_->get_observation_history().size(), 5u);

  std::vector<float> action = {0.5f};
  for (size_t i = 0; i < 10; ++i) {
    preprocessor_->set_previous_actions(action);
  }
  // Should only keep 3 actions
  EXPECT_EQ(preprocessor_->get_action_history().size(), 3u);
}

TEST_F(PreprocessorCoreTest, RegisterObservationProviderSuccess)
{
  EXPECT_FALSE(preprocessor_->has_observation_providers());

  std::vector<float> obs_values = {1.0f, 2.0f, 3.0f};
  rclcpp::Time timestamp(1000, 0);
  ObservationData obs_data = observation_data_from_floats(obs_values, timestamp);

  preprocessor_->register_observation_provider(
    "test_provider",
    {"obs1", "obs2", "obs3"},
    [&obs_data]() -> const ObservationData & {return obs_data;});

  EXPECT_TRUE(preprocessor_->has_observation_providers());
}

TEST_F(PreprocessorCoreTest, RegisterDuplicateObservationProviderThrows)
{
  std::vector<float> obs_values = {1.0f};
  rclcpp::Time timestamp(1000, 0);
  ObservationData obs_data = observation_data_from_floats(obs_values, timestamp);

  preprocessor_->register_observation_provider(
    "duplicate_name",
    {"obs1"},
    [&obs_data]() -> const ObservationData & {return obs_data;});

  EXPECT_THROW(
    preprocessor_->register_observation_provider(
      "duplicate_name",
      {"obs1"},
      [&obs_data]() -> const ObservationData & {return obs_data;}),
    std::runtime_error);
}

TEST_F(PreprocessorCoreTest, ClearObservationProviders)
{
  std::vector<float> obs_values = {1.0f};
  rclcpp::Time timestamp(1000, 0);
  ObservationData obs_data = observation_data_from_floats(obs_values, timestamp);

  preprocessor_->register_observation_provider(
    "provider1",
    {"obs1"},
    [&obs_data]() -> const ObservationData & {return obs_data;});

  EXPECT_TRUE(preprocessor_->has_observation_providers());
  preprocessor_->clear_observation_providers();
  EXPECT_FALSE(preprocessor_->has_observation_providers());
}

TEST_F(PreprocessorCoreTest, BuildObservationSuccess)
{
  std::vector<float> obs_values1 = {1.0f, 2.0f};
  std::vector<float> obs_values2 = {3.0f, 4.0f, 5.0f};
  rclcpp::Time timestamp1(900, 0);
  rclcpp::Time timestamp2(950, 0);
  ObservationData obs_data1 = observation_data_from_floats(obs_values1, timestamp1);
  ObservationData obs_data2 = observation_data_from_floats(obs_values2, timestamp2);

  preprocessor_->register_observation_provider(
    "provider1",
    {"a", "b"},
    [&obs_data1]() -> const ObservationData & {return obs_data1;});

  preprocessor_->register_observation_provider(
    "provider2",
    {"c", "d", "e"},
    [&obs_data2]() -> const ObservationData & {return obs_data2;});

  rclcpp::Time current_time(1000, 0);
  EXPECT_TRUE(preprocessor_->build_observation(current_time));

  const auto & observation = preprocessor_->get_observation();
  ASSERT_EQ(observation.size(), 5u);
  EXPECT_FLOAT_EQ(observation[0], 1.0f);
  EXPECT_FLOAT_EQ(observation[1], 2.0f);
  EXPECT_FLOAT_EQ(observation[2], 3.0f);
  EXPECT_FLOAT_EQ(observation[3], 4.0f);
  EXPECT_FLOAT_EQ(observation[4], 5.0f);

  // Verify time diffs
  const auto & time_diffs = preprocessor_->get_observation_time_diffs();
  ASSERT_EQ(time_diffs.size(), 2u);
  ASSERT_EQ(time_diffs.count("provider1"), 1u);
  ASSERT_EQ(time_diffs.count("provider2"), 1u);
  EXPECT_DOUBLE_EQ(time_diffs.at("provider1"), 100.0);
  EXPECT_DOUBLE_EQ(time_diffs.at("provider2"), 50.0);

  const auto & list = preprocessor_->get_observation_named_value_list();
  ASSERT_EQ(list.size(), 1u);
  EXPECT_EQ(list[0].name, "observation");
  ASSERT_TRUE(list[0].value.is_tensor());
  EXPECT_EQ(list[0].value.as_tensor().num_elements(), 5u);
}

TEST_F(PreprocessorCoreTest, BuildObservationWithTensorValueMatchesFlatVector)
{
  auto storage = std::make_shared<std::vector<float>>(
    std::initializer_list<float>{2.0f, 3.0f});
  const Value obs_tensor(Tensor::share_vector(storage, {2}));
  rclcpp::Time timestamp(900, 0);
  ObservationData obs_data(obs_tensor, timestamp);

  preprocessor_->register_observation_provider(
    "tensor_provider",
    {"a", "b"},
    [&obs_data]() -> const ObservationData & {return obs_data;});

  rclcpp::Time current_time(1000, 0);
  ASSERT_TRUE(preprocessor_->build_observation(current_time));

  const auto & flat = preprocessor_->get_observation();
  ASSERT_EQ(flat.size(), 2u);
  EXPECT_FLOAT_EQ(flat[0], 2.0f);
  EXPECT_FLOAT_EQ(flat[1], 3.0f);

  const auto & list = preprocessor_->get_observation_named_value_list();
  ASSERT_EQ(list.size(), 1u);
  EXPECT_EQ(list[0].name, "observation");
  ASSERT_TRUE(list[0].value.is_tensor());
  const Tensor & out_tensor = list[0].value.as_tensor();
  ASSERT_EQ(out_tensor.num_elements(), 2u);
  const auto span = out_tensor.span<float>();
  EXPECT_FLOAT_EQ(span[0], 2.0f);
  EXPECT_FLOAT_EQ(span[1], 3.0f);
}

TEST_F(PreprocessorCoreTest, BuildObservationThrowsOnEmptyVector)
{
  std::vector<float> empty_values;
  rclcpp::Time timestamp(900, 0);
  ObservationData obs_data = observation_data_from_floats(std::move(empty_values), timestamp);

  preprocessor_->register_observation_provider(
    "empty_provider",
    {},
    [&obs_data]() -> const ObservationData & {return obs_data;});

  rclcpp::Time current_time(1000, 0);
  EXPECT_THROW(preprocessor_->build_observation(current_time), std::runtime_error);
}

TEST_F(PreprocessorCoreTest, BuildObservationThrowsOnFutureTimestamp)
{
  std::vector<float> obs_values = {1.0f};
  rclcpp::Time future_timestamp(2000, 0);
  ObservationData obs_data = observation_data_from_floats(obs_values, future_timestamp);

  preprocessor_->register_observation_provider(
    "future_provider",
    {"obs1"},
    [&obs_data]() -> const ObservationData & {return obs_data;});

  rclcpp::Time current_time(1000, 0);
  EXPECT_THROW(preprocessor_->build_observation(current_time), std::runtime_error);
}

TEST_F(PreprocessorCoreTest, BuildObservationThrowsOnSizeMismatch)
{
  std::vector<float> obs_values = {1.0f, 2.0f};  // 2 values
  rclcpp::Time timestamp(900, 0);
  ObservationData obs_data = observation_data_from_floats(obs_values, timestamp);

  preprocessor_->register_observation_provider(
    "mismatched_provider",
    {"obs1", "obs2", "obs3"},  // 3 segment names
    [&obs_data]() -> const ObservationData & {return obs_data;});

  rclcpp::Time current_time(1000, 0);
  EXPECT_THROW(preprocessor_->build_observation(current_time), std::runtime_error);
}

TEST_F(PreprocessorCoreTest, BuildObservationThrowsOnEmptyValue)
{
  rclcpp::Time timestamp(900, 0);
  ObservationData obs_data(Value(), timestamp);

  preprocessor_->register_observation_provider(
    "empty_value_provider",
    {"a"},
    [&obs_data]() -> const ObservationData & {return obs_data;});

  rclcpp::Time current_time(1000, 0);
  EXPECT_THROW(preprocessor_->build_observation(current_time), std::runtime_error);
}

TEST_F(PreprocessorCoreTest, BuildObservationThrowsOnNonFloat32Tensor)
{
  auto storage = std::make_shared<std::vector<int32_t>>(std::vector<int32_t>{7});
  const Value v(Tensor::share_vector(storage, {1}));
  rclcpp::Time timestamp(900, 0);
  ObservationData obs_data(v, timestamp);

  preprocessor_->register_observation_provider(
    "int32_provider",
    {"seg"},
    [&obs_data]() -> const ObservationData & {return obs_data;});

  rclcpp::Time current_time(1000, 0);
  EXPECT_THROW(preprocessor_->build_observation(current_time), std::runtime_error);
}

TEST_F(PreprocessorCoreTest, TensorRejectsNonEmptyStrides)
{
  auto storage = std::make_shared<std::vector<float>>(std::vector<float>{1.0f, 2.0f});
  EXPECT_THROW(
    Tensor(
      DataType::Float32,
      {2},
      ByteBufferView::share_vector(storage),
      {},
      {1}),
    std::invalid_argument);
}

TEST_F(PreprocessorCoreTest, GetObservationNames)
{
  std::vector<float> obs_values1 = {1.0f, 2.0f};
  std::vector<float> obs_values2 = {3.0f};
  rclcpp::Time timestamp(900, 0);
  ObservationData obs_data1 = observation_data_from_floats(obs_values1, timestamp);
  ObservationData obs_data2 = observation_data_from_floats(obs_values2, timestamp);

  preprocessor_->register_observation_provider(
    "provider1",
    {"joint1_pos", "joint2_pos"},
    [&obs_data1]() -> const ObservationData & {return obs_data1;});

  preprocessor_->register_observation_provider(
    "provider2",
    {"velocity"},
    [&obs_data2]() -> const ObservationData & {return obs_data2;});

  auto names = preprocessor_->get_observation_names();
  ASSERT_EQ(names.size(), 3u);
  EXPECT_EQ(names[0], "joint1_pos");
  EXPECT_EQ(names[1], "joint2_pos");
  EXPECT_EQ(names[2], "velocity");
}

TEST_F(PreprocessorCoreTest, SetPreviousObservationsWithZeroHistoryLength)
{
  // Default config has observation_history_length = 0
  PreprocessorCoreConfig config;
  config.observation_history_length = 0;
  preprocessor_->set_config(config);

  std::vector<float> obs = {1.0f, 2.0f};
  preprocessor_->set_previous_observations(obs);

  // With zero history length, nothing should be stored
  EXPECT_EQ(preprocessor_->get_observation_history().size(), 0u);
}

TEST_F(PreprocessorCoreTest, SetPreviousObservationsThrowsOnSizeMismatch)
{
  PreprocessorCoreConfig config;
  config.observation_history_length = 5;
  preprocessor_->set_config(config);

  std::vector<float> obs1 = {1.0f, 2.0f};
  preprocessor_->set_previous_observations(obs1);

  std::vector<float> obs2 = {1.0f, 2.0f, 3.0f};  // Different size
  EXPECT_THROW(preprocessor_->set_previous_observations(obs2), std::runtime_error);
}

TEST_F(PreprocessorCoreTest, SetPreviousActionsWithZeroHistoryLength)
{
  // Default config has action_history_length = 0
  PreprocessorCoreConfig config;
  config.action_history_length = 0;
  preprocessor_->set_config(config);

  std::vector<float> action = {0.5f};
  preprocessor_->set_previous_actions(action);

  // With zero history length, nothing should be stored
  EXPECT_EQ(preprocessor_->get_action_history().size(), 0u);
}

TEST_F(PreprocessorCoreTest, SetPreviousActionsThrowsOnSizeMismatch)
{
  PreprocessorCoreConfig config;
  config.action_history_length = 5;
  preprocessor_->set_config(config);

  std::vector<float> action1 = {0.5f};
  preprocessor_->set_previous_actions(action1);

  std::vector<float> action2 = {0.5f, 0.6f};  // Different size
  EXPECT_THROW(preprocessor_->set_previous_actions(action2), std::runtime_error);
}

TEST_F(PreprocessorCoreTest, HistoryOrderMostRecentFirst)
{
  PreprocessorCoreConfig config;
  config.observation_history_length = 3;
  config.action_history_length = 3;
  preprocessor_->set_config(config);

  // Add observations in order 1, 2, 3
  preprocessor_->set_previous_observations({1.0f});
  preprocessor_->set_previous_observations({2.0f});
  preprocessor_->set_previous_observations({3.0f});

  const auto & obs_history = preprocessor_->get_observation_history();
  ASSERT_EQ(obs_history.size(), 3u);
  // Most recent should be first
  EXPECT_FLOAT_EQ(obs_history[0][0], 3.0f);
  EXPECT_FLOAT_EQ(obs_history[1][0], 2.0f);
  EXPECT_FLOAT_EQ(obs_history[2][0], 1.0f);

  // Same for actions
  preprocessor_->set_previous_actions({10.0f});
  preprocessor_->set_previous_actions({20.0f});
  preprocessor_->set_previous_actions({30.0f});

  const auto & action_history = preprocessor_->get_action_history();
  ASSERT_EQ(action_history.size(), 3u);
  EXPECT_FLOAT_EQ(action_history[0][0], 30.0f);
  EXPECT_FLOAT_EQ(action_history[1][0], 20.0f);
  EXPECT_FLOAT_EQ(action_history[2][0], 10.0f);
}

TEST_F(PreprocessorCoreTest, HistoryPopsOldestWhenFull)
{
  PreprocessorCoreConfig config;
  config.observation_history_length = 2;
  preprocessor_->set_config(config);

  preprocessor_->set_previous_observations({1.0f});
  preprocessor_->set_previous_observations({2.0f});
  preprocessor_->set_previous_observations({3.0f});  // Should pop {1.0f}

  const auto & obs_history = preprocessor_->get_observation_history();
  ASSERT_EQ(obs_history.size(), 2u);
  EXPECT_FLOAT_EQ(obs_history[0][0], 3.0f);
  EXPECT_FLOAT_EQ(obs_history[1][0], 2.0f);
}

TEST_F(PreprocessorCoreTest, BuildObservationClearsAndRebuilds)
{
  auto obs_storage = std::make_shared<std::vector<float>>(std::vector<float>{1.0f});
  rclcpp::Time timestamp(900, 0);
  ObservationData obs_data = observation_data_from_float_vector(obs_storage, timestamp);

  preprocessor_->register_observation_provider(
    "provider",
    {"obs"},
    [&obs_data]() -> const ObservationData & {return obs_data;});

  rclcpp::Time current_time(1000, 0);
  preprocessor_->build_observation(current_time);
  EXPECT_FLOAT_EQ(preprocessor_->get_observation()[0], 1.0f);

  (*obs_storage)[0] = 100.0f;
  preprocessor_->build_observation(current_time);
  EXPECT_FLOAT_EQ(preprocessor_->get_observation()[0], 100.0f);
}

}  // namespace ros2_policy_execution_core
