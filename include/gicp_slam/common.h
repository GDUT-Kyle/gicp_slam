#ifndef _COMMON_H_
#define _COMMON_H_

#include <iostream>
#include <string>
#include <cmath>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <ros/ros.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>	//	pcl::transformPointCloud 用到这个头文件
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>

#include <eigen3/Eigen/Dense>

#include "livox_ros_driver/CustomMsg.h"

using namespace std;

#define rad2deg 180.0/M_PI
#define deg2rad M_PI/180.0

struct Transform{
    double timestamp;
    Eigen::Vector3d transition;
    Eigen::Quaterniond rotation;
    Transform(){
        timestamp = 0.0;
        transition.setZero();
        rotation.setIdentity();
    }
    Transform(double ts, Eigen::Vector3d t, Eigen::Quaterniond q):timestamp(ts), transition(t), rotation(q){}
};

struct pointCloudWithTime{
    using  Point = pcl::PointXYZI;
    using  PointCloud = pcl::PointCloud<Point>;
    double timestamp;
    PointCloud pointcloud;
    pointCloudWithTime()
    {
        timestamp = 0.0;
        pointcloud.clear();
    }
};

#endif  // _COMMON_H_