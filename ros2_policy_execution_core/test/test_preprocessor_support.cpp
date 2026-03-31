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

#include <vector>

#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"

#include "ros2_policy_execution_core/preprocessor_support.hpp"

namespace ros2_policy_execution_core
{

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

  std::vector<float> v1 = {1.0f};
  std::vector<float> v2 = {2.0f, 3.0f};
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

  EXPECT_FLOAT_EQ(registry.providers()[0].second().values[0], 1.0f);
  ASSERT_EQ(registry.providers()[1].second().values.size(), 2u);
  EXPECT_FLOAT_EQ(registry.providers()[1].second().values[0], 2.0f);
}

TEST(HistoryManagerTest, ObservationAndActionLengthsIndependent)
{
  HistoryManager hm;
  hm.set_lengths(2, 1);
  hm.push_observation({1.0f});
  hm.push_observation({2.0f});
  hm.push_action({9.0f});
  ASSERT_EQ(hm.observations().size(), 2u);
  ASSERT_EQ(hm.actions().size(), 1u);
  EXPECT_FLOAT_EQ(hm.actions()[0][0], 9.0f);
}

TEST(HistoryManagerTest, SetLengthsToZeroClearsBuffers)
{
  HistoryManager hm;
  hm.set_lengths(5, 5);
  hm.push_observation({1.0f});
  hm.push_action({2.0f});
  hm.set_lengths(0, 0);
  EXPECT_TRUE(hm.observations().empty());
  EXPECT_TRUE(hm.actions().empty());
}

}  // namespace ros2_policy_execution_core
