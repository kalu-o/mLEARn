project(mlearn) # the project name
cmake_minimum_required(VERSION 3.5.1) # 
set(Boost_USE_STATIC_LIBS   ON)
set(CMAKE_CXX_STANDARD 11) # Use c++11.
# Set build type if not set.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
# Set additional flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror  -Wpedantic -D NDEBUG -DBOOST_UBLAS_NDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}  -static -O3 -Wall -Werror  -Wpedantic -D NDEBUG -DBOOST_UBLAS_NDEBUG   -lboost_program_options -lboost_serialization")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Werror -g -O0")
# directory for binary output, libries and *.h files:
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(INCLUDE_DIRECTORY_PATH ${PROJECT_SOURCE_DIR}/include)
include_directories(${INCLUDE_DIRECTORY_PATH})
link_directories (/usr/lib)
find_package(Boost)
if(Boost_FOUND)
    message("Boost Found")
    include_directories("${INCLUDE_DIRECTORY_PATH} ${Boost_INCLUDE_DIRS}")
elseif(NOT Boost_FOUND)
    error("Boost Not Found. Boost libries required. sudo apt-get install libboost-all-dev")
endif()

add_library(util STATIC   src/libutil.cpp)
target_link_libraries(util PUBLIC  boost_serialization boost_program_options)
add_library(layer STATIC  src/layer.cpp)
target_link_libraries(layer PUBLIC util)
add_library(network STATIC  src/network.cpp)
target_link_libraries(network PUBLIC  layer)
add_library(data_reader STATIC  src/data_reader.cpp)
target_link_libraries(data_reader PUBLIC util)
add_library(optimizer STATIC  src/optimizer.cpp)
target_link_libraries(optimizer PUBLIC network)
add_library(core STATIC  src/optimizer.cpp)
target_link_libraries(core PUBLIC network data_reader)
# add executable mlearn
add_executable(mlearn  src/main.cpp)

target_link_libraries(mlearn  data_reader optimizer)
add_subdirectory(tests)

