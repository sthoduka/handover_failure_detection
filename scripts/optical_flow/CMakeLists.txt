cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
find_package(OpenCV REQUIRED)

find_package(Boost 1.46 COMPONENTS
    filesystem
    system
    REQUIRED
)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

add_executable(calculate_optical_flow
    src/calculate_optical_flow.cpp
)

target_link_libraries(calculate_optical_flow
    ${Boost_LIBRARIES}
    ${OpenCV_LIBS}
)
