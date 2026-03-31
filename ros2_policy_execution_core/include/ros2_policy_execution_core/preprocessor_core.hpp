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

#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rclcpp/time.hpp"

namespace ros2_policy_execution_core
{

struct PreprocessorCoreConfig
{
  // Add any configuration parameters needed for the preprocessor here
  size_t observation_history_length = 0;
  size_t action_history_length = 0;
};

/**
 * @brief Data returned by an observation provider.
 *
 * Contains both the observation values and an optional timestamp.
 */
struct ObservationData
{
  const std::vector<float> & values;  ///< Reference to observation values
  rclcpp::Time timestamp;              ///< Timestamp of the observation data
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
  /// @brief Function signature for observation data providers
  using ObservationProvider = std::function<const ObservationData &()>;

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
   * @brief Register an observation provider with a name.
   *
   * Providers are called in order of registration when build_observation() is invoked.
   * The vectors returned by each provider are concatenated in registration order
   * to form the final observation vector.
   *
   * @param[in] name Unique name for this segment (used for debugging/introspection)
   * @param[in] provider Function that returns a vector of floats for this segment
   * @throws std::runtime_error if a provider with the same name already exists
   */
  void register_observation_provider(const std::string & name, ObservationProvider provider)
  {
    // Check for duplicate names
    for (const auto & entry : observation_providers_)
    {
      if (entry.first == name)
      {
        throw std::runtime_error("Observation provider with name '" + name + "' already exists.");
      }
    }
    observation_providers_.emplace_back(name, std::move(provider));
  }

  /**
   * @brief Build the observation vector by calling all registered providers.
   *
   * Providers are called in the order they were registered, and their outputs
   * are concatenated to form the final observation vector. The time difference
   * between the provided current_time and each observation's timestamp is stored.
   *
   * @param[in] current_time The current time to calculate time differences against
   * @return true if observation was built successfully, false otherwise
   */
  virtual bool build_observation(const rclcpp::Time & current_time)
  {
    current_observation_.clear();
    observation_time_diffs_.clear();
    for (const auto & [name, provider] : observation_providers_)
    {
      const auto & data = provider();
      current_observation_.insert(
        current_observation_.end(), data.values.begin(), data.values.end());
      observation_time_diffs_[name] = (current_time - data.timestamp).seconds();
    }
    return true;
  }

  /**
   * @brief Get all observation provider time differences.
   *
   * @return const reference to the map of provider names to time differences (in seconds)
   */
  [[nodiscard]] const std::unordered_map<std::string, double> &
  get_observation_time_diffs() const
  {
    return observation_time_diffs_;
  }

  /**
   * @brief Check if any observation providers are registered.
   *
   * @return true if at least one provider is registered
   */
  [[nodiscard]] bool has_observation_providers() const
  {
    return !observation_providers_.empty();
  }

  /**
   * @brief Clear all registered observation providers.
   */
  void clear_observation_providers()
  {
    observation_providers_.clear();
  }

  /**
   * @brief Get the current observation vector.
   *
   * Returns the observation vector, whether it was set via set_observation_data()
   * or built via build_observation().
   *
   * @return const reference to the observation vector.
   */
  [[nodiscard]] virtual const std::vector<float> & get_observation() const
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
  [[nodiscard]] const std::deque<std::vector<float>> & get_observation_history() const
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
  [[nodiscard]] const std::deque<std::vector<float>> & get_action_history() const
  {
    return actions_;
  }

  void set_previous_observations(const std::vector<float> & observations)
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

  void set_previous_actions(const std::vector<float> & actions)
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
  std::vector<float> current_observation_ = {};

  /// Storage for observation data as a deque of vectors of floats
  std::deque<std::vector<float>> observations_ = {};

  /// Storage for action data as a deque of vectors of floats
  std::deque<std::vector<float>> actions_ = {};

  /// Length of the observation vector
  size_t observation_history_length_ = 0;

  /// Length of the action vector
  size_t action_history_length_ = 0;

  /// Registered observation providers (name -> provider function)
  std::vector<std::pair<std::string, ObservationProvider>> observation_providers_ = {};

  /// Time differences per observation provider (name -> seconds since observation timestamp)
  std::unordered_map<std::string, double> observation_time_diffs_ = {};
};

}  // namespace ros2_policy_execution_core

#endif  // ROS2_POLICY_EXECUTION_CORE__PREPROCESSOR_CORE_HPP_
