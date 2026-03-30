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

#ifndef ROS2_POLICY_EXECUTION_CORE__PREPROCESSOR_CORE_HPP_
#define ROS2_POLICY_EXECUTION_CORE__PREPROCESSOR_CORE_HPP_

#include <deque>
#include <exception>
#include <stdexcept>
#include <vector>

namespace ros2_policy_execution_core
{

struct PreprocessorCoreConfig
{
  // Add any configuration parameters needed for the preprocessor here
  size_t observation_history_length = 0;
  size_t action_history_length = 0;
};

/**
 * @brief Abstract base class for preprocessors in the policy execution pipeline.
 *
 * This class serves as a plugin base class for creating custom preprocessors
 * that transform raw observations and actions into a format suitable for policy execution.
 */
class PreprocessorCore
{
public:
  /**
   * @brief Virtual destructor for proper cleanup of derived classes.
   */
  virtual ~PreprocessorCore() = default;

  /**
   * @brief Set the configuration for the preprocessor.
   *
   * @param config The configuration object containing observation and action lengths.
   */
  void set_config(const PreprocessorCoreConfig & config)
  {
    observation_history_length_ = config.observation_history_length;
    action_history_length_ = config.action_history_length;
  }

  /**
   * @brief Set the current observation vector.
   *
   * This pure virtual method must be implemented by derived classes to store
   * the current observation vector and manage the history of observations.
   *
   * @param[in] observation The current observation vector to be stored.
   */
  virtual void set_observation_data(const std::vector<double> & observation)
  {
    current_observation_ = observation;
  }

  /**
   * @brief Build and return the observation vector.
   *
   * This pure virtual method must be implemented by derived classes to construct
   * the observation vector from the stored data.
   *
   * @return const reference to the observation vector.
   */
  virtual [[nodiscard]] const std::vector<double> & get_observation() const
  {
    return current_observation_;
  }

  /**
   * @brief Return the history of observations
   * @note the first element of the history is the most recent observation, and 
   * the last element is the oldest observation
   * 
   * @return const reference to the vector of observation vectors.
   */
  [[nodiscard]] const std::deque<std::vector<double>> & get_observation_history() const
  {
    return observations_;
  }

  /**
   * @brief Return the history of actions
   * @note the first element of the history is the most recent action, and 
   * the last element is the oldest action
   * 
   * @return const reference to the vector of action vectors.
   */
  [[nodiscard]] const std::deque<std::vector<double>> & get_action_history() const
  {
    return actions_;
  }

  void set_previous_observations(const std::vector<double> & observations)
  {
    if (observation_history_length_ > 0)
    {
      if (!observations_.empty() && observations_[0].size() != observations.size())
      {
        throw std::runtime_error("Observation size does not match the previous observation size.");
      }
      if (observations_.size() >= observation_history_length_)
      {
          observations_.pop_back();
      }
      observations_.push_front(observations);
    }
  }

  void set_previous_actions(const std::vector<double> & actions)
  {
    if (action_history_length_ > 0)
    {
      if (!actions_.empty() && actions_[0].size() != actions.size())
      {
        throw std::runtime_error("Action size does not match the previous action size.");
      }
      if (actions_.size() >= action_history_length_)
      {
          actions_.pop_back();
      }
      actions_.push_front(actions);
    }
  }

private:
  /// Storage for the current observation vector
  std::vector<double> current_observation_ = {};

  /// Storage for observation data as a deque of vectors of doubles
  std::deque<std::vector<double>> observations_ = {};

  /// Storage for action data as a deque of vectors of doubles
  std::deque<std::vector<double>> actions_ = {};

  /// Length of the observation vector
  size_t observation_history_length_ = 0;

  /// Length of the action vector
  size_t action_history_length_ = 0;
};

}  // namespace ros2_policy_execution_core

#endif  // ROS2_POLICY_EXECUTION_CORE__PREPROCESSOR_CORE_HPP_
