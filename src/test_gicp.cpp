#include<iostream>
#include <chrono>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/gicp.h>
#include "gicp.hpp"
using namespace std;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "mygicp");
    ros::NodeHandle nh("~");

    std::string target_pcd;
    std::string source_pcd;
    nh.param<string>("source", source_pcd, "");
    nh.param<string>("target", target_pcd, "");

    double tx, ty, tz, rx, ry, rz;
    nh.param<double>("tx", tx, 0.0);
    nh.param<double>("ty", ty, 0.0);
    nh.param<double>("tz", tz, 0.0);
    nh.param<double>("rx", rx, 0.0);
    nh.param<double>("ry", ry, 0.0);
    nh.param<double>("rz", rz, 0.0);

    Eigen::Isometry3d Trans = Eigen::Isometry3d::Identity();
    Eigen::AngleAxisd init_rz(rz, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd init_ry(ry, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd init_rx(rx, Eigen::Vector3d::UnitX());
    Trans.rotate(init_rz*init_ry*init_rx);
    Eigen::Vector3d init_translation(tx, ty, tz);
    Trans.pretranslate(init_translation);

    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>());

    if(pcl::io::loadPCDFile(target_pcd, *target_cloud)) {
        std::cerr << "failed to load " << target_pcd << std::endl;
        return 0;
    }
    if(pcl::io::loadPCDFile(source_pcd, *source_cloud)) {
        std::cerr << "failed to load " << source_pcd << std::endl;
        return 0;
    }

    // downsampling
    pcl::PointCloud<pcl::PointXYZ>::Ptr ds_source_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr ds_target_cloud(new pcl::PointCloud<pcl::PointXYZ>());

    pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
    voxelgrid.setLeafSize(0.2f, 0.2f, 0.2f);

    voxelgrid.setInputCloud(target_cloud);
    voxelgrid.filter(*ds_target_cloud);
    // *target_cloud = *downsampled;

    voxelgrid.setInputCloud(source_cloud);
    voxelgrid.filter(*ds_source_cloud);
    // source_cloud = downsampled;

    std::cout<<"downsampled_source_cloud size = "<<ds_source_cloud->size()<<endl;
    std::cout<<"downsampled_target_cloud size = "<<ds_target_cloud->size()<<endl;

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>::Ptr pclgicp(new pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>());
    pclgicp->setInputSource(source_cloud);
    pclgicp->setInputTarget(ds_target_cloud);
    pclgicp->setMaxCorrespondenceDistance(5.0);
    pclgicp->setMaximumIterations(20);
    pclgicp->setMaximumOptimizerIterations(10);
    pclgicp->setRotationEpsilon(1e-3);
    pclgicp->setTransformationEpsilon(1e-3);
    pclgicp->align(*aligned_cloud, Trans.matrix().cast<float>());

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    ROS_INFO("solve time cost = %f seconds.", time_used1.count());

    cout<<pclgicp->getFinalTransformation()<<endl;

    pcl::transformPointCloud(*source_cloud, *aligned_cloud, pclgicp->getFinalTransformation());

    // visualization
    pcl::visualization::PCLVisualizer vis1("pcl-gicp");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_handler1(target_cloud, 255.0, 0.0, 0.0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_handler1(source_cloud, 0.0, 0.0, 255.0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> aligned_handler1(aligned_cloud, 0.0, 255.0, 0.0);
    vis1.addPointCloud(target_cloud, target_handler1, "target");
    vis1.addPointCloud(source_cloud, source_handler1, "source");
    vis1.addPointCloud(aligned_cloud, aligned_handler1, "aligned");
    // vis1.spin();

    // --------------------------------------------------------------------------------
    cout<<"--------------------------------------------------------------------------------"<<endl;

    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

    gicp::GeneralizedICP<pcl::PointXYZ>::Ptr gicper(new gicp::GeneralizedICP<pcl::PointXYZ>());

    gicper->setInnerParameterTolerance(1e-6);
    gicper->setInnerGradientTolerance(1e-6);
    gicper->setOptimizationEpsilon(1e-3);

    gicper->setInputSource(ds_source_cloud);
    gicper->setInputTarget(ds_target_cloud);
    
    gicper->align(Trans.matrix(), aligned_cloud);

    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used2 = std::chrono::duration_cast<std::chrono::duration<double>>(t4 - t3);
    ROS_INFO("solve time cost = %f seconds.", time_used2.count());

    cout<<gicper->getTransformation()<<endl;

    pcl::transformPointCloud(*source_cloud, *aligned_cloud, gicper->getTransformation());

    // visualization
    pcl::visualization::PCLVisualizer vis2("cvte-gicp");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_handler2(target_cloud, 255.0, 0.0, 0.0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_handler2(source_cloud, 0.0, 0.0, 255.0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> aligned_handler2(aligned_cloud, 0.0, 255.0, 0.0);
    vis2.addPointCloud(target_cloud, target_handler2, "target");
    vis2.addPointCloud(source_cloud, source_handler2, "source");
    vis2.addPointCloud(aligned_cloud, aligned_handler2, "aligned");
    vis2.spin();

    return 0;
}