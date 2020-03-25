cmake_minimum_required(VERSION 3.5)

project(convectionDiffusion LANGUAGES CXX)

set(CMAKE_CXX_FLAGS -fopenmp)
set(CMAKE_CXX_FLAGS -O3)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER "/home/omarkahol/.openmpi/bin/mpic++")

FIND_PACKAGE(deal.II 9.1.0 REQUIRED
  HINTS ${DEAL_II_DIR} /home/omarkahol/.deal.II $ENV{DEAL_II_DIR}
  )
DEAL_II_INITIALIZE_CACHED_VARIABLES()
add_executable(convectionDiffusion main.cpp)

DEAL_II_SETUP_TARGET(convectionDiffusion)

