cmake_minimum_required(VERSION 3.8)
project(ros2_policy_execution_core)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

option(ROS2_POLICY_EXECUTION_CORE_BUILD_EXAMPLES "Build example executables" ON)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

set(ONNXRUNTIME_ROOT "/opt/onnxruntime" CACHE PATH "Root directory of the ONNX Runtime installation")
find_path(
  ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT}/include
)
find_library(
  ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  HINTS ${ONNXRUNTIME_ROOT}/lib
)

set(HAVE_ONNXRUNTIME FALSE)
if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
  set(HAVE_ONNXRUNTIME TRUE)
  add_library(onnxruntime_local INTERFACE)
  target_include_directories(onnxruntime_local INTERFACE ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(onnxruntime_local INTERFACE ${ONNXRUNTIME_LIBRARY})
else()
  message(WARNING "ONNX Runtime was not found. ORT examples and tests will be skipped.")
endif()

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)
target_link_libraries(${PROJECT_NAME} INTERFACE
  pluginlib::pluginlib
  rclcpp::rclcpp)

if(ROS2_POLICY_EXECUTION_CORE_BUILD_EXAMPLES)
  add_executable(
    ros2_preprocessor_value_example
    examples/ros2_preprocessor_value_example.cpp
  )
  target_link_libraries(ros2_preprocessor_value_example ${PROJECT_NAME})
  ament_target_dependencies(
    ros2_preprocessor_value_example
    geometry_msgs
    rclcpp
    sensor_msgs
  )

  install(
    TARGETS ros2_preprocessor_value_example
    DESTINATION lib/${PROJECT_NAME}
  )

  if(HAVE_ONNXRUNTIME)
    add_executable(
      value_types_basic_example
      examples/value_types_basic_example.cpp
    )
    target_link_libraries(
      value_types_basic_example
      ${PROJECT_NAME}
      onnxruntime_local
    )

    install(
      TARGETS value_types_basic_example
      DESTINATION lib/${PROJECT_NAME}
    )
  endif()
endif()

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

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gmock REQUIRED)
  ament_lint_auto_find_test_dependencies()

  ament_add_gmock(test_preprocessor
    test/test_preprocessor_core.cpp
    test/test_preprocessor_support.cpp
  )
  target_link_libraries(test_preprocessor
    ${PROJECT_NAME}
  )

  ament_add_gmock(test_value_types
    test/test_value_types.cpp
  )
  target_link_libraries(test_value_types
    ${PROJECT_NAME}
  )

  if(HAVE_ONNXRUNTIME)
    ament_add_gmock(test_ort_value_conversion
      test/test_ort_value_conversion.cpp
    )
    target_link_libraries(
      test_ort_value_conversion
      ${PROJECT_NAME}
      onnxruntime_local
    )
  endif()
endif()

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(pluginlib rclcpp)

ament_package()
