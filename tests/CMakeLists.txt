cmake_minimum_required(VERSION 3.5.1)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
add_executable(unit_tests  unit_tests_suite.cpp)
target_link_libraries(unit_tests  data_readers optimizers)
