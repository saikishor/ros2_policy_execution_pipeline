# Copyright 2026 PAI SIG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#[=======================================================================[.rst:
FindONNXRUNTIME
---------------

Find the ONNX Runtime headers and shared library.

Result variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``ONNXRUNTIME_FOUND``
  True if ONNX Runtime was found.
``ONNXRUNTIME_INCLUDE_DIR``
  Directory containing ``onnxruntime_cxx_api.h``.
``ONNXRUNTIME_LIBRARY``
  Path to the ONNX Runtime shared library.
``ONNXRUNTIME_VERSION``
  Version string read from ``VERSION_NUMBER`` when available.

Imported targets
^^^^^^^^^^^^^^^^

This module defines the following imported target:

``ONNXRUNTIME::ONNXRUNTIME``
  Imported target for the ONNX Runtime library.

Hints
^^^^^

You may set ``ONNXRUNTIME_ROOT`` or the environment variable
``ONNXRUNTIME_ROOT`` to point at a custom installation prefix.
``/opt/onnxruntime`` is also searched as an additional common path.
#]=======================================================================]

include(FindPackageHandleStandardArgs)

set(_onnxruntime_roots "")
if(ONNXRUNTIME_ROOT)
  list(APPEND _onnxruntime_roots "${ONNXRUNTIME_ROOT}")
endif()
if(DEFINED ENV{ONNXRUNTIME_ROOT})
  list(APPEND _onnxruntime_roots "$ENV{ONNXRUNTIME_ROOT}")
endif()
list(APPEND _onnxruntime_roots "/opt/onnxruntime")

find_path(
  ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS ${_onnxruntime_roots}
  PATH_SUFFIXES include
)

find_library(
  ONNXRUNTIME_LIBRARY
  NAMES onnxruntime libonnxruntime
  HINTS ${_onnxruntime_roots}
  PATH_SUFFIXES lib lib64
)

find_file(
  ONNXRUNTIME_VERSION_FILE
  NAMES VERSION_NUMBER
  HINTS ${_onnxruntime_roots}
)

if(ONNXRUNTIME_VERSION_FILE)
  file(STRINGS "${ONNXRUNTIME_VERSION_FILE}" ONNXRUNTIME_VERSION LIMIT_COUNT 1)
endif()

find_package_handle_standard_args(
  ONNXRUNTIME
  REQUIRED_VARS ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIBRARY
  VERSION_VAR ONNXRUNTIME_VERSION
)

if(ONNXRUNTIME_FOUND AND NOT TARGET ONNXRUNTIME::ONNXRUNTIME)
  add_library(ONNXRUNTIME::ONNXRUNTIME UNKNOWN IMPORTED)
  set_target_properties(ONNXRUNTIME::ONNXRUNTIME PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
  )
endif()

mark_as_advanced(
  ONNXRUNTIME_INCLUDE_DIR
  ONNXRUNTIME_LIBRARY
  ONNXRUNTIME_VERSION_FILE
)
