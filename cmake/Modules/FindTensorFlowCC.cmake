INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_TENSORFLOW_CC tensorflow_cc)

if(PC_TENSORFLOW_CC_FOUND)
    # look for include files
    FIND_PATH(
        TENSORFLOW_CC_INCLUDE_DIR
        NAMES tensorflow/cc/client/client_session.h
        HINTS $ENV{TENSORFLOW_CC_DIR}/include
              ${PC_TENSORFLOW_CC_INCLUDE_DIRS}
              ${CMAKE_INSTALL_PREFIX}/include
        PATHS /usr/local/include
              /usr/include
    )
    # look for libs
    FIND_LIBRARY(
        TENSORFLOW_CC_LIBRARIES
        NAMES tensorflow_cc
        HINTS $ENV{TENSORFLOW_CC_DIR}/lib
              ${PC_TENSORFLOW_CC_LIBDIR}
              ${CMAKE_INSTALL_PREFIX}/lib/
              ${CMAKE_INSTALL_PREFIX}/lib64/
        PATHS /usr/local/lib
              /usr/local/lib64
              /usr/lib
              /usr/lib64
    )
    set(TENSORFLOW_CC_FOUND ${PC_TENSORFLOW_CC_FOUND})
endif(PC_TENSORFLOW_CC_FOUND)

set(TENSORFLOW_CC_INCLUDE_DIRS "${TENSORFLOW_CC_INCLUDE_DIR}; "
    "${TENSORFLOW_CC_INCLUDE_DIR}/tensorflow "
    "${TENSORFLOW_CC_INCLUDE_DIR}/third_party "
    "${TENSORFLOW_CC_INCLUDE_DIR}/eigen "
    "${TENSORFLOW_CC_INCLUDE_DIR}/google/protobuf "
    )

message(STATUS "Tensorflow_cc include dirs: ")
message(STATUS ${TENSORFLOW_CC_INCLUDE_DIRS})

INCLUDE(FindPackageHandleStandardArgs)
# do not check TENSORFLOW_CC_INCLUDE_DIRS, is not set when default include path is used.
FIND_PACKAGE_HANDLE_STANDARD_ARGS(TENSORFLOW_CC DEFAULT_MSG TENSORFLOW_CC_LIBRARIES)
MARK_AS_ADVANCED(TENSORFLOW_CC_LIBRARIES TENSORFLOW_CC_INCLUDE_DIRS)
