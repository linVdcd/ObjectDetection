

cmake_minimum_required(VERSION 3.4.1)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O2 -fvisibility=hidden -fopenmp -lz -llog -ljnigraphics -fvisibility-inlines-hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math")
OPTION(ANDROID "build android library" TRUE)


add_definitions(-DANDROID_OPEN_FILE)
add_library( # Sets the name of the library.
             squeezencnn

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             src/main/jni/squeezencnn_jni.cpp )

find_library(
          android-lib
          android
          )

find_library(
              log-lib
              log )

add_library(
    ncnn
    STATIC
    IMPORTED
    )
set_target_properties(
    ncnn
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/libs/${ANDROID_ABI}/libncnn.a
    )


include_directories(src/main/jni/include)
add_library(android_fs STATIC ../../android_fs/android_fs.c)
include_directories(../../android_fs/)
add_library(darknet_region STATIC src/main/jni/region_layer.cpp src/main/jni/box.cpp )
target_link_libraries(
                       squeezencnn
                       android_fs
                       ncnn
                       darknet_region
                       ${log-lib}
                       ${android-lib}
                       )