cmake_minimum_required(VERSION 3.8)
project(ros2_policy_execution_pipeline)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(onnxruntime_vendor REQUIRED)
find_package(ros2_policy_execution_core REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/policy_execution_pipeline.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
target_link_libraries(${PROJECT_NAME} PUBLIC
  pluginlib::pluginlib
  rclcpp::rclcpp
  onnxruntime::onnxruntime
  ros2_policy_execution_core::ros2_policy_execution_core)

add_executable(policy_execution_pipeline_node src/policy_execution_pipeline_node.cpp)
target_link_libraries(policy_execution_pipeline_node PRIVATE
  ${PROJECT_NAME})

install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install the executable to lib/<package_name> so `ros2 run` and launch files can find it
install(
  TARGETS policy_execution_pipeline_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(pluginlib rclcpp onnxruntime_vendor ros2_policy_execution_core)

ament_package()
