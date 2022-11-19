cmake_minimum_required(VERSION 3.0.2)
project(gicp_slam)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  nav_msgs
  rosbag
  roscpp
  sensor_msgs
  std_msgs
  tf
  livox_ros_driver
  rosbag
)

# see here: https://github.com/PointCloudLibrary/pcl/issues/3680
# when this is fixed, we can remove the following 3 lines.
if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()
find_package(PCL REQUIRED)

find_package(Eigen3 3.3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(
  include
	${catkin_INCLUDE_DIRS} 
	${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS})

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp std_msgs
  DEPENDS EIGEN3 PCL
  INCLUDE_DIRS include
)

add_executable(test_gicp src/test_gicp.cpp)
target_link_libraries(test_gicp ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(gicp_odometry src/gicp_odometry.cpp)
target_link_libraries(gicp_odometry ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} OpenMP::OpenMP_CXX)


