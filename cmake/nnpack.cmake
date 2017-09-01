if (__NNPACK_INCLUDED)
  return()
endif()
set(__NNPACK_INCLUDED TRUE)
 
if (NOT USE_NNPACK)
  return()
endif()

# try any external nnpack first



##############################################################################
# Custom build rules to build nnpack, if external dependency is not found 
##############################################################################

set(NNPACK_PREFIX ${NNPACK_ROOT})

##############################################################################
# (1) MSVC - unsupported 
##############################################################################

if (MSVC)
  message(WARNING "NNPACK not supported on MSVC yet. Turn this warning off by USE_NNPACK=OFF.")
  set(USE_NNPACK OFF)
  return()
endif()

##############################################################################
# (2) Mobile platform - direct build
##############################################################################

# We know that NNPACK is only supported with clang due to some builtin
# function usages, so we will guard it.
#if (ANDROID)
#  if (NOT ${CMAKE_C_COMPILER_ID} STREQUAL "Clang")
#    message(WARNING
#            "NNPACK currently requires the clang compiler to build. Seems "
#            "that we are not building with clang, so I will turn off NNPACK "
#            "build.")
#    set(USE_NNPACK OFF)
#    return()
#  endif()
#endif()

if (ANDROID)
  include(${NNPACK_ROOT}/android-cmake/AndroidNdkModules.cmake)
  android_ndk_import_module_cpufeatures()
endif()


if (ANDROID OR IOS)
  message(WARNING "NNPACK for mobile cmake support is wip")
  

  # pthreadpool
  set(CAFFE2_PTHREADPOOL_SRCS
      ${NNPACK_ROOT}/deps/pthreadpool/src/threadpool-pthreads.c)
  add_library(CAFFE2_PTHREADPOOL STATIC ${CAFFE2_PTHREADPOOL_SRCS})
  target_include_directories(CAFFE2_PTHREADPOOL PRIVATE
      ${NNPACK_ROOT}/deps/fxdiv/include
      ${NNPACK_ROOT}/deps/pthreadpool/include)
  # nnpack
  set(CAFFE2_NNPACK_SRCS
      # nnpack_ukernels: common files
      ${NNPACK_ROOT}/src/psimd/2d-fourier-8x8.c
      ${NNPACK_ROOT}/src/psimd/2d-fourier-16x16.c
      ${NNPACK_ROOT}/src/psimd/2d-winograd-8x8-3x3.c
      ${NNPACK_ROOT}/src/psimd/relu.c
      ${NNPACK_ROOT}/src/psimd/softmax.c
      ${NNPACK_ROOT}/src/psimd/fft-block-mac.c
      ${NNPACK_ROOT}/src/psimd/blas/shdotxf.c
      # nnpack_ukernels: neon files
      ${NNPACK_ROOT}/src/neon/blas/conv1x1.c
      ${NNPACK_ROOT}/src/neon/blas/s4gemm.c
      ${NNPACK_ROOT}/src/neon/blas/c4gemm.c
      ${NNPACK_ROOT}/src/neon/blas/s4c2gemm.c
      ${NNPACK_ROOT}/src/neon/blas/c4gemm-conjb.c
      ${NNPACK_ROOT}/src/neon/blas/s4c2gemm-conjb.c
      ${NNPACK_ROOT}/src/neon/blas/c4gemm-conjb-transc.c
      ${NNPACK_ROOT}/src/neon/blas/s4c2gemm-conjb-transc.c
      ${NNPACK_ROOT}/src/neon/blas/sgemm.c
      ${NNPACK_ROOT}/src/neon/blas/sdotxf.c
      # nnpack files
      ${NNPACK_ROOT}/src/init.c
      ${NNPACK_ROOT}/src/convolution-output.c
      ${NNPACK_ROOT}/src/convolution-input-gradient.c
      ${NNPACK_ROOT}/src/convolution-inference.c
      ${NNPACK_ROOT}/src/fully-connected-output.c
      ${NNPACK_ROOT}/src/fully-connected-inference.c
      ${NNPACK_ROOT}/src/pooling-output.c
      ${NNPACK_ROOT}/src/softmax-output.c
      ${NNPACK_ROOT}/src/relu-output.c
      ${NNPACK_ROOT}/src/relu-input-gradient.c
  )
  add_library(CAFFE2_NNPACK STATIC ${CAFFE2_NNPACK_SRCS})
  target_include_directories(CAFFE2_NNPACK PRIVATE
      ${NNPACK_ROOT}/include
      ${NNPACK_ROOT}/src
      ${NNPACK_ROOT}/deps/fp16/include
      ${NNPACK_ROOT}/deps/fxdiv/include
      ${NNPACK_ROOT}/deps/pthreadpool/include
      ${NNPACK_ROOT}/deps/psimd/include)

  set(NNPACK_FOUND TRUE)
  # nnpack.h itself only needs to have nnpack and pthreadpool as include directories.
  set(NNPACK_INCLUDE_DIRS ${NNPACK_ROOT}/deps/fp16/include ${NNPACK_ROOT}/deps/fxdiv/include
      ${NNPACK_ROOT}/include
      ${NNPACK_ROOT}/deps/pthreadpool/include)
  set(NNPACK_LIBRARIES $<TARGET_FILE:CAFFE2_NNPACK> $<TARGET_FILE:CAFFE2_PTHREADPOOL>)
  if (ANDROID)
    set(NNPACK_LIBRARIES ${NNPACK_LIBRARIES} cpufeatures)
  endif()
  return()
endif()

##############################################################################
# (3) Linux/Mac: use PeachPy
##############################################################################

if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux" OR ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  message(STATUS "Will try to build NNPACK from source. If anything fails, "
                 "follow the NNPACK prerequisite installation steps.")
  find_program(CAFFE2_CONFU_COMMAND confu)
  find_program(CAFFE2_NINJA_COMMAND ninja)
  if (CAFFE2_CONFU_COMMAND AND CAFFE2_NINJA_COMMAND)
    # Note: per Marat, there is no support for fPIC right now so we will need to
    # manually change it in build.ninja
    ExternalProject_Add(nnpack_external
        SOURCE_DIR ${NNPACK_PREFIX}
        BUILD_IN_SOURCE 1
        CONFIGURE_COMMAND ""
        BUILD_COMMAND confu setup
        COMMAND python ./configure.py
        COMMAND sed -ibuild.ninja.bak "s/cflags = /cflags = -fPIC /" build.ninja
        COMMAND sed -ibuild.ninja.bak "s/cxxflags = /cxxflags = -fPIC /" build.ninja
        COMMAND ninja nnpack
        INSTALL_COMMAND ""
        )

    set(NNPACK_FOUND TRUE)
    set(NNPACK_INCLUDE_DIRS
        ${NNPACK_PREFIX}/include
        ${NNPACK_PREFIX}/deps/pthreadpool/include)
    set(NNPACK_LIBRARIES ${NNPACK_PREFIX}/lib/libnnpack.a ${NNPACK_PREFIX}/lib/libpthreadpool.a)
    list(APPEND Caffe2_EXTERNAL_DEPENDENCIES nnpack_external)
  else()
    message(WARNING "NNPACK is chosen to be installed, but confu and ninja "
                    "that are needed by it are not installed. As a result "
                    "we won't build with NNPACK.")
    set(USE_NNPACK OFF)
  endif()
  return()
endif()

##############################################################################
# (3) Catch-all: not supported.
##############################################################################

message(WARNING "Unknown platform - I don't know how to build NNPACK. "
                "See cmake/External/nnpack.cmake for details.")
set(USE_NNPACK OFF)
