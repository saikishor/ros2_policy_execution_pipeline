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

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "onnxruntime_cxx_api.h"
#include "rclcpp/rclcpp.hpp"

#include "ros2_policy_execution_core/preprocessor_core.hpp"

namespace ros2_policy_execution_core
{

namespace
{
/// Creates one 1-D single-element Ort::Value tensor per float.
/// IMPORTANT: `data` must outlive the returned vector.
std::vector<Ort::Value> make_ort_values(std::vector<float> & data)
{
  static Ort::MemoryInfo mem_info =
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<int64_t> shape = {1};
  std::vector<Ort::Value> values;
  values.reserve(data.size());
  for (float & v : data) {
    values.push_back(
      Ort::Value::CreateTensor<float>(mem_info, &v, 1, shape.data(), 1));
  }
  return values;
}
}  // namespace

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
  std::vector<float> obs_raw = {1.0f, 2.0f};
  auto obs_ort = make_ort_values(obs_raw);
  for (size_t i = 0; i < 10; ++i) {
    preprocessor_->set_previous_observations(obs_ort);
  }
  // Should only keep 5 observations
  EXPECT_EQ(preprocessor_->get_observation_history().size(), 5u);

  std::vector<float> action_raw = {0.5f};
  auto action_ort = make_ort_values(action_raw);
  for (size_t i = 0; i < 10; ++i) {
    preprocessor_->set_previous_actions(action_ort);
  }
  // Should only keep 3 actions
  EXPECT_EQ(preprocessor_->get_action_history().size(), 3u);
}

TEST_F(PreprocessorCoreTest, RegisterObservationProviderSuccess)
{
  EXPECT_FALSE(preprocessor_->has_observation_providers());

  std::vector<float> obs_raw = {1.0f, 2.0f, 3.0f};
  auto obs_ort = make_ort_values(obs_raw);
  rclcpp::Time timestamp(1000, 0);
  ObservationData obs_data{obs_ort, timestamp};

  preprocessor_->register_observation_provider(
    "test_provider",
    {"obs1", "obs2", "obs3"},
    [&obs_data]() -> const ObservationData & {return obs_data;});

  EXPECT_TRUE(preprocessor_->has_observation_providers());
}

TEST_F(PreprocessorCoreTest, RegisterDuplicateObservationProviderThrows)
{
  std::vector<float> obs_raw = {1.0f};
  auto obs_ort = make_ort_values(obs_raw);
  rclcpp::Time timestamp(1000, 0);
  ObservationData obs_data{obs_ort, timestamp};

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
  std::vector<float> obs_raw = {1.0f};
  auto obs_ort = make_ort_values(obs_raw);
  rclcpp::Time timestamp(1000, 0);
  ObservationData obs_data{obs_ort, timestamp};

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
  std::vector<float> raw1 = {1.0f, 2.0f};
  std::vector<float> raw2 = {3.0f, 4.0f, 5.0f};
  auto ort1 = make_ort_values(raw1);
  auto ort2 = make_ort_values(raw2);
  rclcpp::Time timestamp1(900, 0);
  rclcpp::Time timestamp2(950, 0);
  ObservationData obs_data1{ort1, timestamp1};
  ObservationData obs_data2{ort2, timestamp2};

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
  EXPECT_FLOAT_EQ(*observation[0].GetTensorData<float>(), 1.0f);
  EXPECT_FLOAT_EQ(*observation[1].GetTensorData<float>(), 2.0f);
  EXPECT_FLOAT_EQ(*observation[2].GetTensorData<float>(), 3.0f);
  EXPECT_FLOAT_EQ(*observation[3].GetTensorData<float>(), 4.0f);
  EXPECT_FLOAT_EQ(*observation[4].GetTensorData<float>(), 5.0f);

  // Verify time diffs
  const auto & time_diffs = preprocessor_->get_observation_time_diffs();
  ASSERT_EQ(time_diffs.size(), 2u);
  ASSERT_EQ(time_diffs.count("provider1"), 1u);
  ASSERT_EQ(time_diffs.count("provider2"), 1u);
  EXPECT_DOUBLE_EQ(time_diffs.at("provider1"), 100.0);
  EXPECT_DOUBLE_EQ(time_diffs.at("provider2"), 50.0);
}

TEST_F(PreprocessorCoreTest, BuildObservationThrowsOnEmptyVector)
{
  std::vector<float> empty_raw;
  auto empty_ort = make_ort_values(empty_raw);
  rclcpp::Time timestamp(900, 0);
  ObservationData obs_data{empty_ort, timestamp};

  preprocessor_->register_observation_provider(
    "empty_provider",
    {},
    [&obs_data]() -> const ObservationData & {return obs_data;});

  rclcpp::Time current_time(1000, 0);
  EXPECT_THROW(preprocessor_->build_observation(current_time), std::runtime_error);
}

TEST_F(PreprocessorCoreTest, BuildObservationThrowsOnFutureTimestamp)
{
  std::vector<float> obs_raw = {1.0f};
  auto obs_ort = make_ort_values(obs_raw);
  rclcpp::Time future_timestamp(2000, 0);
  ObservationData obs_data{obs_ort, future_timestamp};

  preprocessor_->register_observation_provider(
    "future_provider",
    {"obs1"},
    [&obs_data]() -> const ObservationData & {return obs_data;});

  rclcpp::Time current_time(1000, 0);
  EXPECT_THROW(preprocessor_->build_observation(current_time), std::runtime_error);
}

TEST_F(PreprocessorCoreTest, BuildObservationThrowsOnSizeMismatch)
{
  std::vector<float> obs_raw = {1.0f, 2.0f};  // 2 values
  auto obs_ort = make_ort_values(obs_raw);
  rclcpp::Time timestamp(900, 0);
  ObservationData obs_data{obs_ort, timestamp};

  preprocessor_->register_observation_provider(
    "mismatched_provider",
    {"obs1", "obs2", "obs3"},  // 3 segment names
    [&obs_data]() -> const ObservationData & {return obs_data;});

  rclcpp::Time current_time(1000, 0);
  EXPECT_THROW(preprocessor_->build_observation(current_time), std::runtime_error);
}

TEST_F(PreprocessorCoreTest, GetObservationNames)
{
  std::vector<float> raw1 = {1.0f, 2.0f};
  std::vector<float> raw2 = {3.0f};
  auto ort1 = make_ort_values(raw1);
  auto ort2 = make_ort_values(raw2);
  rclcpp::Time timestamp(900, 0);
  ObservationData obs_data1{ort1, timestamp};
  ObservationData obs_data2{ort2, timestamp};

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

  std::vector<float> obs_raw = {1.0f, 2.0f};
  auto obs_ort = make_ort_values(obs_raw);
  preprocessor_->set_previous_observations(obs_ort);

  // With zero history length, nothing should be stored
  EXPECT_EQ(preprocessor_->get_observation_history().size(), 0u);
}

TEST_F(PreprocessorCoreTest, SetPreviousObservationsThrowsOnSizeMismatch)
{
  PreprocessorCoreConfig config;
  config.observation_history_length = 5;
  preprocessor_->set_config(config);

  std::vector<float> obs1_raw = {1.0f, 2.0f};
  auto obs1_ort = make_ort_values(obs1_raw);
  preprocessor_->set_previous_observations(obs1_ort);

  std::vector<float> obs2_raw = {1.0f, 2.0f, 3.0f};  // Different size
  auto obs2_ort = make_ort_values(obs2_raw);
  EXPECT_THROW(preprocessor_->set_previous_observations(obs2_ort), std::runtime_error);
}

TEST_F(PreprocessorCoreTest, SetPreviousActionsWithZeroHistoryLength)
{
  // Default config has action_history_length = 0
  PreprocessorCoreConfig config;
  config.action_history_length = 0;
  preprocessor_->set_config(config);

  std::vector<float> action_raw = {0.5f};
  auto action_ort = make_ort_values(action_raw);
  preprocessor_->set_previous_actions(action_ort);

  // With zero history length, nothing should be stored
  EXPECT_EQ(preprocessor_->get_action_history().size(), 0u);
}

TEST_F(PreprocessorCoreTest, SetPreviousActionsThrowsOnSizeMismatch)
{
  PreprocessorCoreConfig config;
  config.action_history_length = 5;
  preprocessor_->set_config(config);

  std::vector<float> action1_raw = {0.5f};
  auto action1_ort = make_ort_values(action1_raw);
  preprocessor_->set_previous_actions(action1_ort);

  std::vector<float> action2_raw = {0.5f, 0.6f};  // Different size
  auto action2_ort = make_ort_values(action2_raw);
  EXPECT_THROW(preprocessor_->set_previous_actions(action2_ort), std::runtime_error);
}

TEST_F(PreprocessorCoreTest, HistoryOrderMostRecentFirst)
{
  PreprocessorCoreConfig config;
  config.observation_history_length = 3;
  config.action_history_length = 3;
  preprocessor_->set_config(config);

  // Add observations in order 1, 2, 3
  std::vector<float> obs1_raw = {1.0f};
  std::vector<float> obs2_raw = {2.0f};
  std::vector<float> obs3_raw = {3.0f};
  auto obs1_ort = make_ort_values(obs1_raw);
  auto obs2_ort = make_ort_values(obs2_raw);
  auto obs3_ort = make_ort_values(obs3_raw);
  preprocessor_->set_previous_observations(obs1_ort);
  preprocessor_->set_previous_observations(obs2_ort);
  preprocessor_->set_previous_observations(obs3_ort);

  const auto & obs_history = preprocessor_->get_observation_history();
  ASSERT_EQ(obs_history.size(), 3u);
  // Most recent should be first
  EXPECT_FLOAT_EQ(*obs_history[0][0].GetTensorData<float>(), 3.0f);
  EXPECT_FLOAT_EQ(*obs_history[1][0].GetTensorData<float>(), 2.0f);
  EXPECT_FLOAT_EQ(*obs_history[2][0].GetTensorData<float>(), 1.0f);

  // Same for actions
  std::vector<float> act1_raw = {10.0f};
  std::vector<float> act2_raw = {20.0f};
  std::vector<float> act3_raw = {30.0f};
  auto act1_ort = make_ort_values(act1_raw);
  auto act2_ort = make_ort_values(act2_raw);
  auto act3_ort = make_ort_values(act3_raw);
  preprocessor_->set_previous_actions(act1_ort);
  preprocessor_->set_previous_actions(act2_ort);
  preprocessor_->set_previous_actions(act3_ort);

  const auto & action_history = preprocessor_->get_action_history();
  ASSERT_EQ(action_history.size(), 3u);
  EXPECT_FLOAT_EQ(*action_history[0][0].GetTensorData<float>(), 30.0f);
  EXPECT_FLOAT_EQ(*action_history[1][0].GetTensorData<float>(), 20.0f);
  EXPECT_FLOAT_EQ(*action_history[2][0].GetTensorData<float>(), 10.0f);
}

TEST_F(PreprocessorCoreTest, HistoryPopsOldestWhenFull)
{
  PreprocessorCoreConfig config;
  config.observation_history_length = 2;
  preprocessor_->set_config(config);

  std::vector<float> obs1_raw = {1.0f};
  std::vector<float> obs2_raw = {2.0f};
  std::vector<float> obs3_raw = {3.0f};
  auto obs1_ort = make_ort_values(obs1_raw);
  auto obs2_ort = make_ort_values(obs2_raw);
  auto obs3_ort = make_ort_values(obs3_raw);
  preprocessor_->set_previous_observations(obs1_ort);
  preprocessor_->set_previous_observations(obs2_ort);
  preprocessor_->set_previous_observations(obs3_ort);  // Should pop obs1

  const auto & obs_history = preprocessor_->get_observation_history();
  ASSERT_EQ(obs_history.size(), 2u);
  EXPECT_FLOAT_EQ(*obs_history[0][0].GetTensorData<float>(), 3.0f);
  EXPECT_FLOAT_EQ(*obs_history[1][0].GetTensorData<float>(), 2.0f);
}

TEST_F(PreprocessorCoreTest, BuildObservationClearsAndRebuilds)
{
  std::vector<float> obs_raw = {1.0f};
  auto obs_ort = make_ort_values(obs_raw);
  rclcpp::Time timestamp(900, 0);
  ObservationData obs_data{obs_ort, timestamp};

  preprocessor_->register_observation_provider(
    "provider",
    {"obs"},
    [&obs_data]() -> const ObservationData & {return obs_data;});

  rclcpp::Time current_time(1000, 0);
  preprocessor_->build_observation(current_time);
  EXPECT_FLOAT_EQ(*preprocessor_->get_observation()[0].GetTensorData<float>(), 1.0f);

  // Change the values in the underlying buffer (obs_ort tensors wrap obs_raw directly)
  obs_raw[0] = 100.0f;
  preprocessor_->build_observation(current_time);
  EXPECT_FLOAT_EQ(*preprocessor_->get_observation()[0].GetTensorData<float>(), 100.0f);
}

}  // namespace ros2_policy_execution_core
