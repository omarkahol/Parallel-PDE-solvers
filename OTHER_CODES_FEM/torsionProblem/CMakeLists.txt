cmake_minimum_required(VERSION 3.5)

project(torsionProblem LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED C++14)
set(CMAKE_CXX_COMPILER "/home/omarkahol/.openmpi/bin/mpicxx")
set(CMAKE_CXX_FLAGS -O3)

FIND_PACKAGE(deal.II 9.1.0 REQUIRED
  HINTS ${DEAL_II_DIR} /home/omarkahol/.deal2 $ENV{DEAL_II_DIR}
  )
DEAL_II_INITIALIZE_CACHED_VARIABLES()
add_executable(torsionProblem main.cpp)

DEAL_II_SETUP_TARGET(torsionProblem)
