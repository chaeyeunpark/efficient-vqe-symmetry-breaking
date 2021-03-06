cmake_minimum_required(VERSION 3.9)

project(VQE_TI)

set(YAVQUE_SOURCE_DIR "Yavque")
include_directories("${YAVQUE_SOURCE_DIR}/include")
include_directories("${YAVQUE_SOURCE_DIR}/externals/nlohmann_json_cmake_fetchcontent/include/")
include_directories("${YAVQUE_SOURCE_DIR}/externals") 
include_directories("${YAVQUE_SOURCE_DIR}/externals/ExactDiagonalization/Include/")
add_subdirectory("./Yavque")

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

enable_testing()

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -DEIGEN_USE_MKL_ALL")


## Setup MKL
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	include_directories("$ENV{MKLROOT}/include")
	link_libraries(mkl_intel_lp64 mkl_tbb_thread mkl_core tbb pthread m dl)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	include_directories("$ENV{MKLROOT}/include")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs")
	link_libraries(mkl_intel_lp64 mkl_tbb_thread mkl_core tbb pthread m dl)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl=parallel -tbb")
	include_directories("$ENV{MKLROOT}/include")
	link_libraries(tbb stdc++ pthread m dl)
endif()

link_directories(${CMAKE_SOURCE_DIR}/lib)

add_executable(tfi "src/tfi.cpp")
target_link_libraries(tfi qunn)

add_executable(tfi_brksym "src/tfi_brksym.cpp")
target_link_libraries(tfi_brksym qunn)

add_executable(tfi_brksym_incdepth "src/tfi_brksym_incdepth.cpp")
target_link_libraries(tfi_brksym_incdepth qunn)

add_executable(cluster "src/cluster.cpp")
target_link_libraries(cluster qunn)

add_executable(cluster_brksym "src/cluster_brksym.cpp")
target_link_libraries(cluster_brksym qunn)

add_executable(cluster_stab "src/cluster_stab.cpp")
target_link_libraries(cluster_stab qunn)

