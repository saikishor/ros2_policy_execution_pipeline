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

#ifndef ROS2_POLICY_EXECUTION_CORE__PREPROCESSOR_SUPPORT_HPP_
#define ROS2_POLICY_EXECUTION_CORE__PREPROCESSOR_SUPPORT_HPP_

#include <deque>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <onnxruntime_cxx_api.h>

#include "rclcpp/time.hpp"

namespace ros2_policy_execution_core
{

/**
 * @brief Data returned by an observation provider.
 *
 * Contains both the observation values and an optional timestamp.
 */
struct ObservationData
{
  const std::vector<std::shared_ptr<Ort::Value>> & values;  ///< Reference to observation values
  rclcpp::Time timestamp;                                    ///< Timestamp of the observation data
};

/**
 * @brief Registry for observation providers and their segment names.
 *
 * Holds providers in registration order with parallel segment name lists for use by the preprocessor.
 */
class ObservationProviderRegistry
{
public:
  /// @brief Function signature for observation data providers.
  using ObservationProvider = std::function<const ObservationData &()>;

  /**
   * @brief Register a named provider and its segment names.
   *
   * @param[in] name Unique name for this provider (used for debugging and time-diff maps).
   * @param[in] observation_segment_names Names of individual value segments from this provider.
   * @param[in] provider Callable that returns observation values and timestamp.
   * @throws std::runtime_error if a provider with the same name already exists.
   */
  void register_provider(
    const std::string & name,
    const std::vector<std::string> & observation_segment_names,
    ObservationProvider provider)
  {
    for (const auto & entry : providers_) {
      if (entry.first == name) {
        throw std::runtime_error("Observation provider with name '" + name + "' already exists.");
      }
    }
    providers_.emplace_back(name, std::move(provider));
    segment_names_.emplace_back(name, observation_segment_names);
  }

  /**
   * @brief Clear all registered providers and segment names.
   */
  void clear()
  {
    providers_.clear();
    segment_names_.clear();
  }

  /**
   * @brief Check if any observation provider is registered.
   *
   * @return true if no providers are registered, false otherwise.
   */
  [[nodiscard]] bool empty() const {return providers_.empty();}

  /**
   * @brief Access registered providers in registration order.
   *
   * @return const reference to the (name, provider) list.
   */
  [[nodiscard]] const std::vector<std::pair<std::string, ObservationProvider>> & providers() const
  {
    return providers_;
  }

  /**
   * @brief Segment name entries parallel to providers().
   *
   * @return const reference to the (provider name, segment names) list.
   */
  [[nodiscard]] const std::vector<std::pair<std::string,
    std::vector<std::string>>> & segment_names() const
  {
    return segment_names_;
  }

private:
  /// Registered observation providers (name and callback), in call order.
  std::vector<std::pair<std::string, ObservationProvider>> providers_ = {};
  /// Segment names per provider, same order as providers_.
  std::vector<std::pair<std::string, std::vector<std::string>>> segment_names_ = {};
};

/**
 * @brief Manages bounded history for observations and actions.
 *
 * When a history length is zero, that history is disabled and the backing deque is cleared.
 * Newest entries are stored at the front of each deque.
 */
class HistoryManager
{
public:
  /**
   * @brief Set observation and action history lengths and trim existing buffers.
   *
   * @param observation_history_length Maximum number of observation snapshots to retain.
   * @param action_history_length Maximum number of action snapshots to retain.
   */
  void set_lengths(size_t observation_history_length, size_t action_history_length)
  {
    observation_history_length_ = observation_history_length;
    action_history_length_ = action_history_length;
    trim_to_length(observations_, observation_history_length_);
    trim_to_length(actions_, action_history_length_);
  }

  /**
   * @brief Push a full observation vector into the observation history.
   *
   * @param[in] observation Vector to store; must match prior entry size when history is non-empty.
   * @throws std::runtime_error if the vector size is inconsistent with existing history.
   */
  void push_observation(const std::vector<std::shared_ptr<Ort::Value>> & observation)
  {
    push_entry(observation, observation_history_length_, observations_, "Observation");
  }

  /**
   * @brief Push a full action vector into the action history.
   *
   * @param[in] action Vector to store; must match prior entry size when history is non-empty.
   * @throws std::runtime_error if the vector size is inconsistent with existing history.
   */
  void push_action(const std::vector<std::shared_ptr<Ort::Value>> & action)
  {
    push_entry(action, action_history_length_, actions_, "Action");
  }

  /**
   * @brief Return the observation history deque.
   * @note The first element is the most recent observation; the last is the oldest.
   *
   * @return const reference to observation history vectors.
   */
  [[nodiscard]] const std::deque<std::vector<std::shared_ptr<Ort::Value>>> & observations() const
  {
    return observations_;
  }

  /**
   * @brief Return the action history deque.
   * @note The first element is the most recent action; the last is the oldest.
   *
   * @return const reference to action history vectors.
   */
  [[nodiscard]] const std::deque<std::vector<std::shared_ptr<Ort::Value>>> & actions() const
  {
    return actions_;
  }

private:
  static void trim_to_length(
    std::deque<std::vector<std::shared_ptr<Ort::Value>>> & data,
    size_t max_length)
  {
    if (max_length == 0) {
      data.clear();
      return;
    }
    while (data.size() > max_length) {
      data.pop_back();
    }
  }

  static void push_entry(
    const std::vector<std::shared_ptr<Ort::Value>> & values, size_t max_length,
    std::deque<std::vector<std::shared_ptr<Ort::Value>>> & history,
    const std::string & value_name)
  {
    if (max_length == 0) {
      return;
    }

    if (!history.empty() && history.front().size() != values.size()) {
      throw std::runtime_error(
              value_name + " size does not match the previous " + value_name + " size.");
    }
    if (history.size() >= max_length) {
      history.pop_back();
    }
    history.push_front(values);
  }

  /// Configured maximum observation history length (0 disables).
  size_t observation_history_length_ = 0;
  /// Configured maximum action history length (0 disables).
  size_t action_history_length_ = 0;
  /// Observation snapshots; front is newest.
  std::deque<std::vector<std::shared_ptr<Ort::Value>>> observations_ = {};
  /// Action snapshots; front is newest.
  std::deque<std::vector<std::shared_ptr<Ort::Value>>> actions_ = {};
};

}  // namespace ros2_policy_execution_core

#endif  // ROS2_POLICY_EXECUTION_CORE__PREPROCESSOR_SUPPORT_HPP_
