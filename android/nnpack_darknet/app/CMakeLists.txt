

cmake_minimum_required(VERSION 3.4.1)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -mfloat-abi=softfp -mfpu=neon -ftree-vectorize -ffast-math")
OPTION(ANDROID "build android library" TRUE)

add_definitions(-DANDROID_OPEN_FILE)
add_library( # Sets the name of the library.
             native-lib

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             src/main/cpp/native-lib.c )

find_library(
          android-lib
          android
          )

find_library(
              log-lib
              log )


#build nnpack
add_definitions("-DNNPACK=1" )
add_definitions("-DARM_NEON=1")
add_definitions("-DHAVE_NEON=1") #多线程+neon 向量矩阵乘法
#add_definitions("-DHAVE_TF_DW=1") #tensorflow depthwise_conv2d,速度比没有任何优化的还慢,因为数据格式不一样，需要reshape

set(root "../../../")
set(NNPACK_ROOT "${root}/NNPACK-darknet")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${NNPACK_ROOT}")
include(${NNPACK_ROOT}/android-cmake/AndroidNdkModules.cmake)
set(USE_NNPACK ON)
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
include("${root}/cmake/nnpack.cmake")
include_directories(${NNPACK_INCLUDE_DIRS})



#build darknet
include_directories("${root}/darknet/include" "${root}/darknet/src" "${root}")
file(GLOB darknet_src ${root}/darknet/src/*.c ${root}/darknet/src/*.cpp)
add_library(darknet ${darknet_src})

#build android fopen to open files in assets dir
add_library(android_fs STATIC ${root}/android/android_fs/android_fs.c)
include_directories( src/main/cpp ${root}/android/android_fs/)




target_link_libraries(
                       native-lib
                       darknet
                       android_fs
                       ${NNPACK_LIBRARIES}

                       ${log-lib}
                       ${android-lib}
                       )