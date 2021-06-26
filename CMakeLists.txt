project(mLEARn) # the project name
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

add_library(utils STATIC   src/utils.cpp)
target_link_libraries(utils PUBLIC  boost_serialization boost_program_options)
add_library(layers STATIC  src/layers.cpp)
target_link_libraries(layers PUBLIC utils)
add_library(networks STATIC  src/networks.cpp)
target_link_libraries(networks PUBLIC  layers)
add_library(data_readers STATIC  src/data_readers.cpp)
target_link_libraries(data_readers PUBLIC utils)
add_library(optimizers STATIC  src/optimizers.cpp)
target_link_libraries(optimizers PUBLIC networks)
add_library(core STATIC  src/optimizers.cpp)
target_link_libraries(core PUBLIC networks data_readers)
## add executable mlearn
add_executable(mlearn  src/main.cpp)
target_link_libraries(mlearn  data_readers optimizers)
add_subdirectory(tests)
#add_subdirectory(misc)

