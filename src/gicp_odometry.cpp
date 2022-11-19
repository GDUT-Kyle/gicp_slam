#include <fstream>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/gicp.h>
#include <chrono>
#include <nav_msgs/Path.h>
#include "gicp.hpp"
#include "gicp_slam/loadBag.h"

class gicp_odometry
{
private:
    ros::NodeHandle nh;

    Eigen::Quaterniond q_w;
    Eigen::Vector3d t_w;

    Eigen::Quaterniond q_rela;
    Eigen::Vector3d t_rela;

    rslidar_loader::Ptr loader;
    std::string bag_path;
    std::string lidar_topic;

    pcl::VoxelGrid<pcl::PointXYZI> voxelgrid;
    pcl::PointCloud<pcl::PointXYZI>::Ptr pointCloudLast;
    pcl::PointCloud<pcl::PointXYZI>::Ptr pointCloudCurr;
    pcl::PointCloud<pcl::PointXYZI>::Ptr aligned_cloud;

    ros::Publisher pubLaserCloudMap;
    ros::Publisher pubPath;

    nav_msgs::Path global_path;

    std::ofstream path_rec;
    std::ofstream time_rec;

    std::string path_rec_file;
    std::string time_rec_file;
public:
    gicp_odometry():nh("~")
    {
        nh.param<std::string>("bag_path", bag_path, "");
        nh.param<std::string>("lidar_topic", lidar_topic, "");
        nh.param<std::string>("path_rec_file", path_rec_file, "");
        nh.param<std::string>("time_rec_file", time_rec_file, "");
        loader.reset(new rslidar_loader());
        loader->loadScansFromROSBag(bag_path, lidar_topic);

        pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 5);
        pubPath = nh.advertise<nav_msgs::Path>("/odom_path", 5);

        q_w.setIdentity();
        t_w.setZero();

        q_rela.setIdentity();
        t_rela.setZero();

        voxelgrid.setLeafSize(0.2f, 0.2f, 0.2f);
        pointCloudLast.reset(new pcl::PointCloud<pcl::PointXYZI>());
        pointCloudCurr.reset(new pcl::PointCloud<pcl::PointXYZI>());
        aligned_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());

        path_rec.open(path_rec_file);
        path_rec.precision(12);
        path_rec.clear();
        time_rec.open(time_rec_file);
        time_rec.precision(12);
        time_rec.clear();
    }
    virtual ~gicp_odometry(){}

    void spin()
    {
        for(int i=0; i<loader->ScanPointClouds.size() && ros::ok(); i++)
        {
            if(i==0)
            {
                voxelgrid.setInputCloud(loader->ScanPointClouds[i].pointcloud.makeShared());
                voxelgrid.filter(*pointCloudLast);
                continue;
            }

            voxelgrid.setInputCloud(loader->ScanPointClouds[i].pointcloud.makeShared());
            voxelgrid.filter(*pointCloudCurr);

            std::cout<<"Point cloud size = "<< pointCloudCurr->size()<<endl;

            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            
            gicp::GeneralizedICP<pcl::PointXYZI>::Ptr gicper(new gicp::GeneralizedICP<pcl::PointXYZI>());
            // 初值比较好的情况可以设置大一点，因为全局最优点就在附近
            gicper->setOptimizationEpsilon(1e-1);
            gicper->setInputTarget(pointCloudLast);
            gicper->setInputSource(pointCloudCurr);
            // std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
            // std::chrono::duration<double> time_used2 = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
            // ROS_INFO("B solve time cost = %f seconds.", time_used2.count());

            Eigen::Matrix4d Trans;
            Trans.setIdentity();
            Trans.block<3, 3>(0, 0) = q_rela.toRotationMatrix();
            Trans.block<3, 1>(0, 3) = t_rela;
            gicper->align(Trans, aligned_cloud);
            q_rela = gicper->getTransformation().block<3, 3>(0, 0);
            t_rela = gicper->getTransformation().block<3, 1>(0, 3);
            // cout<<gicper->getScore()<<endl;

            std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
            std::chrono::duration<double> time_used3 = std::chrono::duration_cast<std::chrono::duration<double>>(t4 - t1);
            ROS_INFO("C solve time cost = %f seconds.", time_used3.count());

            // Eigen::Matrix4d Trans;
            // Trans.setIdentity();
            // Trans.block<3, 3>(0, 0) = q_rela.toRotationMatrix();
            // Trans.block<3, 1>(0, 3) = t_rela;
            // pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI>::Ptr pclgicp(new pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI>());
            // pclgicp->setInputSource(pointCloudCurr);
            // pclgicp->setInputTarget(pointCloudLast);
            // pclgicp->setMaxCorrespondenceDistance(5.0);
            // pclgicp->setMaximumIterations(20);
            // pclgicp->setMaximumOptimizerIterations(4);
            // pclgicp->setRotationEpsilon(1e-3);
            // pclgicp->setTransformationEpsilon(1e-3);
            // pclgicp->align(*aligned_cloud, Trans.cast<float>());
            // q_rela = pclgicp->getFinalTransformation().block<3, 3>(0, 0).cast<double>();
            // t_rela = pclgicp->getFinalTransformation().block<3, 1>(0, 3).cast<double>();

            // std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();
            // std::chrono::duration<double> time_used3 = std::chrono::duration_cast<std::chrono::duration<double>>(t5 - t1);
            // ROS_INFO("C solve time cost = %f seconds.", time_used3.count());

            t_w = t_w + q_w * t_rela;
            q_w = q_w * q_rela;

            Eigen::Matrix4f T_w;
            T_w.setIdentity();
            T_w.block<3, 3>(0, 0) = q_w.toRotationMatrix().cast<float>();
            T_w.block<3, 1>(0, 3) = t_w.cast<float>();

            aligned_cloud->clear();
            pcl::transformPointCloud(*pointCloudCurr, *aligned_cloud, T_w);

            sensor_msgs::PointCloud2 laserCloudAligned;
            pcl::toROSMsg(*aligned_cloud, laserCloudAligned);
            laserCloudAligned.header.stamp = ros::Time().fromSec(loader->ScanPointClouds[i].timestamp);
            laserCloudAligned.header.frame_id = "/map";
            pubLaserCloudMap.publish(laserCloudAligned);

            static tf::TransformBroadcaster br;
			tf::Transform transform;
			tf::Quaternion q;
			transform.setOrigin(tf::Vector3(t_w(0),
											t_w(1),
											t_w(2)));
			q.setW(q_w.w());
			q.setX(q_w.x());
			q.setY(q_w.y());
			q.setZ(q_w.z());
			transform.setRotation(q);
			br.sendTransform(tf::StampedTransform(transform, laserCloudAligned.header.stamp, "/map", "/aft_mapped"));


            *pointCloudLast = *pointCloudCurr;

            geometry_msgs::PoseStamped posemsg;
            posemsg.header.stamp = laserCloudAligned.header.stamp;
            posemsg.header.frame_id = "/aft_mapped";
            posemsg.pose.position.x = t_w(0);
            posemsg.pose.position.y = t_w(1);
            posemsg.pose.position.z = t_w(2);
            posemsg.pose.orientation.x = q_w.x();
            posemsg.pose.orientation.y = q_w.y();
            posemsg.pose.orientation.z = q_w.z();
            posemsg.pose.orientation.w = q_w.w();

            global_path.header.stamp = laserCloudAligned.header.stamp;
            global_path.header.frame_id = "/map";
            global_path.poses.push_back(posemsg);
            pubPath.publish(global_path);

            path_rec<<i<<" "
                    <<t_w(0)<<" "
                    <<t_w(1)<<" "
                    <<t_w(2)<<" "
                    <<q_w.x()<<" "
                    <<q_w.y()<<" "
                    <<q_w.z()<<" "
                    <<q_w.w()<<endl;
            time_rec<<i<<" "
                    <<time_used3.count()<<endl;
        }
        path_rec.close();
        time_rec.close();
    }
};


int main(int argc, char **argv)
{
    ros::init(argc, argv, "mygicp");

    gicp_odometry go;
    go.spin();

    return 0;
}