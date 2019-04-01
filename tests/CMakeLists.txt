cmake_minimum_required(VERSION 3.5.1)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
add_executable(unit_test  unit_test_suite.cpp)
target_link_libraries(unit_test  data_reader optimizer)
