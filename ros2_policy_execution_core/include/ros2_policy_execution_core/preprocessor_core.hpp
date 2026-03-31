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
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rclcpp/node.hpp"
#include "rclcpp/time.hpp"

#include "ros2_policy_execution_core/preprocessor_support.hpp"

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
  /// @brief Function signature for observation data providers
  using ObservationProvider = ObservationProviderRegistry::ObservationProvider;

  /**
   * @brief Virtual destructor for proper cleanup of derived classes.
   */
  virtual ~PreprocessorCore() = default;

  /**
   * @brief Configure the preprocessor with necessary parameters or ROS2 node.
   *
   * This pure virtual method must be implemented by derived classes to perform
   * any necessary configuration steps, such as reading parameters from the ROS2 node.
   * Creating appropriate subscriptions or service clients can also be done here
   *
   * @param[in] node Shared pointer to the ROS2 node for accessing parameters and other resources.
   */
  virtual void configure(const rclcpp::Node::SharedPtr & node) = 0;

  /**
   * @brief Set the configuration for the preprocessor.
   *
   * @param config The configuration object containing observation and action lengths.
   */
  void set_config(const PreprocessorCoreConfig & config)
  {
    history_manager_.set_lengths(
      config.observation_history_length,
      config.action_history_length);
  }

  /**
   * @brief Register an observation provider with a name.
   *
   * Providers are called in order of registration when build_observation() is invoked.
   * The vectors returned by each provider are concatenated in registration order
   * to form the final observation vector.
   *
   * @param[in] name Unique name for this segment (used for debugging/introspection)
   * @param[in] observation_segment_names Names of the individual segments provided by this
   *  provider (used for debugging/introspection)
   * @param[in] provider Function that returns a vector of floats for this segment
   * @throws std::runtime_error if a provider with the same name already exists
   */
  void register_observation_provider(
    const std::string & name,
    const std::vector<std::string> & observation_segment_names,
    ObservationProvider provider)
  {
    provider_registry_.register_provider(name, observation_segment_names, std::move(provider));
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
    const auto & providers = provider_registry_.providers();
    const auto & segment_entries = provider_registry_.segment_names();
    for (size_t i = 0; i < providers.size(); ++i)
    {
      const auto & [name, provider] = providers[i];
      const auto & data = provider();
      const auto & seg_entry = segment_entries[i];
      if (name != seg_entry.first)
      {
        throw std::runtime_error("Observation provider name does not match segment name entry.");
      }
      if (data.values.empty())
      {
        throw std::runtime_error("Observation provider '" + name + "' returned an empty vector.");
      }
      if (data.timestamp > current_time)
      {
        throw std::runtime_error(
                "Observation provider '" + name + "' returned a timestamp in the future.");
      }
      if (data.values.size() != seg_entry.second.size())
      {
        throw std::runtime_error(
                "Observation provider '" + name + "' returned a vector of size " +
                std::to_string(data.values.size()) + " but expected " +
                std::to_string(seg_entry.second.size()) + " based on the segment names.");
      }
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
    return !provider_registry_.empty();
  }

  /**
   * @brief Clear all registered observation providers.
   */
  void clear_observation_providers()
  {
    provider_registry_.clear();
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
    return history_manager_.observations();
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
    return history_manager_.actions();
  }

  void set_previous_observations(const std::vector<float> & observations)
  {
    history_manager_.push_observation(observations);
  }

  void set_previous_actions(const std::vector<float> & actions)
  {
    history_manager_.push_action(actions);
  }

  std::vector<std::string> get_observation_names() const
  {
    std::vector<std::string> segment_names;
    for (const auto & [name, provider_segment_names] : provider_registry_.segment_names())
    {
      segment_names.insert(
        segment_names.end(), provider_segment_names.begin(), provider_segment_names.end());
    }
    return segment_names;
  }

private:
  /// Storage for the current observation vector
  std::vector<float> current_observation_ = {};
  ObservationProviderRegistry provider_registry_;
  HistoryManager history_manager_;

  /// Time differences per observation provider (name -> seconds since observation timestamp)
  std::unordered_map<std::string, double> observation_time_diffs_ = {};
};

}  // namespace ros2_policy_execution_core

#endif  // ROS2_POLICY_EXECUTION_CORE__PREPROCESSOR_CORE_HPP_
