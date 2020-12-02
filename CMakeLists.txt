cmake_minimum_required(VERSION 3.17)
project(Projection CXX)

set(CMAKE_CXX_STANDARD 14)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

# opencv
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# eigen
find_package(Eigen3)
include_directories( ${EIGEN3_INCLUDE_DIRS} )

#Python
find_package(Python2 REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python2_INCLUDE_DIRS})
#find_package(Python2 REQUIRED COMPONENTS Interpreter Development)
#[[set(PYTHON_EXECUTABLE "/usr/bin/python2.7")
set(PYTHON_INCLUDE_DIR "/usr/include/python2.7")
set(PYTHON_LIBRARIES "/usr/lib/x86_64-linux-gnu/libpython2.7.so")]]
#[[message(${Python2_INCLUDE_DIRS})
message(${Python2_LIBRARIES})]]
#message(${Python2_NumPy_INCLUDE_DIRS})
#include_directories(${Python2_INCLUDE_DIRS})
#[[set(PYTHON_INCLUDE_DIRS "/usr/bin/python3.6")
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})
link_directories(/usr/lib/python3.6/config-3.6m-x86_64-linux-gnu)
set(PYTHON_LIBRARIES "/usr/lib/python3.6/config-3.6m-x86_64-linux-gnu/libpython3.6m.so")]]

# Boost
#find_package(Boost 1.68.0 REQUIRED)

# NumCpp
#[[find_package(NumCpp)
include_directories( ${NumCpp_INCLUDE_DIR} )]]

#cnpy
add_subdirectory(cnpy)

# pcl
find_package( PCL REQUIRED COMPONENT common io )
include_directories( ${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions( ${PCL_DEFINITIONS} )

#gflags
find_package( gflags )
include_directories( ${gflags_INCLUDE_DIRS} )
add_definitions( ${gflags_DEFINITIONS} )

add_executable(Projection main.cpp SemanticVoxelGrid.h SemanticVoxelCloud.cpp SemanticVoxelCloud.h ProcessVector.h ProcessVector.cpp voxel_grid.hpp)
target_link_libraries( Projection ${OpenCV_LIBS} )
target_link_libraries( Projection ${PCL_LIBRARIES} -lpcl_filters )
target_link_libraries( Projection gflags )
#target_link_libraries( Projection NumCpp::NumCpp )
target_link_libraries( Projection ${Python3_LIBRARIES} )
target_link_libraries( Projection cnpy )

add_executable(test test.cpp)
#target_link_libraries( Projection NumCpp::NumCpp )
target_link_libraries(test ${Python2_LIBRARIES})
target_link_libraries( test cnpy )