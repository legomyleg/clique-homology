cmake_minimum_required(VERSION 3.24)
project(graphoplib VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)

# EIGEN: Standardizing package inclusion across computers
# Include the FetchContent CMake module
Include(FetchContent)

# Tell CMake where to get Eigen From
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY  https://gitlab.com/libeigen/eigen.git
    GIT_TAG         3.4.0
)

# Make Eigen available to the project, add executable, and link
#   to the target
FetchContent_MakeAvailable(*Eigen3)
add_executable(clique_homology main/main.cpp)
target_link_libraries(clique_homology PRIVATE Eigen3::Eigen)