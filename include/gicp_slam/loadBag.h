#ifndef LOAD_BAG_H
#define LOAD_BAG_H

#include "gicp_slam/common.h"

using namespace std;

class rslidar_loader
{
public:
    typedef shared_ptr<rslidar_loader> Ptr;
    using PointCloudVec = vector<pointCloudWithTime>;
    using PointCloudVecPtr = std::shared_ptr<PointCloudVec>;

    PointCloudVec ScanPointClouds;
public:
    rslidar_loader(){}
    virtual ~rslidar_loader(){}
    bool loadScansFromROSBag(const std::string& bag_path, const std::string topic)
    {
        rosbag::Bag bag;
        try {
            bag.open(bag_path, rosbag::bagmode::Read);
        } catch (rosbag::BagException e) {
            ROS_ERROR_STREAM("LOADING BAG FAILED: " << e.what());
            return false;
        }
        std::vector<std::string> topics;
        topics.push_back(topic);
        
        rosbag::View view(bag, rosbag::TopicQuery(topics));

        // 判断rosbag中的点云帧是否找到
        if (view.size()==0) {
            ROS_ERROR_STREAM(
                "No points were loaded, verify that the bag contains populated "
                "messages of type sensor_msgs/PointCloud2");
            return false;
        }

        // ScanPointClouds.reset(new PointCloudVec(view.size()));
        ScanPointClouds.resize(view.size());

        int ScanPCs = 0;
        for (const rosbag::MessageInstance& m : view) {
            std::cout << " Loading LiDAR scan: \e[1m" << ScanPCs++ << "\e[0m from ros bag"
                    << '\r' << std::flush;

            auto& pointcloud = ScanPointClouds[ScanPCs-1];

            sensor_msgs::PointCloud2 msgPointCloud =
                *(m.instantiate<sensor_msgs::PointCloud2>());

            pcl::fromROSMsg(msgPointCloud, pointcloud.pointcloud);
            pointcloud.timestamp = msgPointCloud.header.stamp.toSec();

            std::vector<int> indices;
            pcl::removeNaNFromPointCloud(pointcloud.pointcloud, pointcloud.pointcloud, indices);
        }
        std::cout << " Loading LiDAR scan: \e[1m" << ScanPCs++ << "\e[0m from ros bag"
                    << '\r' << std::endl;
        return true;
    }
};

class bag_loader{
public:
    // 从ros消息转换至用于操作的数据
    vector<pointCloudWithTime> LivoxPcls;
public:
    bag_loader(){}
    virtual ~bag_loader(){}
    bool loadLivoxsFromROSBag(const std::string& bag_path, const std::string topic, double min_time, double max_time)
    {
        rosbag::Bag bag;
        try {
            bag.open(bag_path, rosbag::bagmode::Read);
        } catch (rosbag::BagException e) {
            ROS_ERROR_STREAM("LOADING BAG FAILED: " << e.what());
            return false;
        }
        std::vector<std::string> topics;
        topics.push_back(topic);
        
        rosbag::View view(bag, rosbag::TopicQuery(topics));

        size_t numLivoxPointClouds = 0;
        for(const rosbag::MessageInstance& m : view)
        {
            if((m.instantiate<livox_ros_driver::CustomMsg>())->header.stamp.toSec()<min_time)
                continue;
            if((m.instantiate<livox_ros_driver::CustomMsg>())->header.stamp.toSec()>max_time)
                break;
            numLivoxPointClouds++;
        }

        // 判断rosbag中的点云帧是否找到
        if (numLivoxPointClouds==0) {
            ROS_ERROR_STREAM(
                "No points were loaded, verify that the bag contains populated "
                "messages of type sensor_msgs/PointCloud2");
            return false;
        }

        LivoxPcls.resize(numLivoxPointClouds);
        
        numLivoxPointClouds=0;
        for (const rosbag::MessageInstance& m : view) {
            if((m.instantiate<livox_ros_driver::CustomMsg>())->header.stamp.toSec()<min_time)
                continue;
            if((m.instantiate<livox_ros_driver::CustomMsg>())->header.stamp.toSec()>max_time)
                break;
            std::cout << " Loading Livox scan: \e[1m" << numLivoxPointClouds++ << "\e[0m from ros bag"
                    << '\r' << std::flush;

            auto& pointcloud = LivoxPcls[numLivoxPointClouds-1];

            // 获取单个点云msg
            livox_ros_driver::CustomMsg LivoxMsg = *(m.instantiate<livox_ros_driver::CustomMsg>());

            // 从msg中提取点云数据
            CustomMsgToPointCloud(LivoxMsg, pointcloud);

            std::vector<int> indices;
            pcl::removeNaNFromPointCloud(pointcloud.pointcloud, pointcloud.pointcloud, indices);
        }
        std::cout << " Loading Livox scan: \e[1m" << numLivoxPointClouds++ << "\e[0m from ros bag"
                    << '\r' << std::endl;
        return true;
        // 至此，rosbag中的点云帧按时间戳排列放到scans_中
    }

    void CustomMsgToPointCloud(const livox_ros_driver::CustomMsg& CustomMsg, pointCloudWithTime& PointCloud)
    {
        // 通过引用，方便操作每一帧
        auto& livox_msg = CustomMsg;
        // 获取该帧最后一个点的相对时间
        auto time_end = livox_msg.points.back().offset_time;

        PointCloud.timestamp = CustomMsg.header.stamp.toSec();
        // 重新组织成PCL的点云
        for (unsigned int i = 0; i < livox_msg.point_num; ++i) {
            pointCloudWithTime::Point pt;
            pt.x = livox_msg.points[i].x;
            pt.y = livox_msg.points[i].y;
            pt.z = livox_msg.points[i].z;
        //      if (pt.z < -0.3) continue; // delete some outliers (our Horizon's assembly height is 0.3 meters)
            float s = livox_msg.points[i].offset_time / (float)time_end;
        //       ROS_INFO("_s-------- %.6f ",s);
            // 线数存在整数部分，时间偏移存在
            pt.intensity = livox_msg.points[i].line + 0.1*s; // The integer part is line number and the decimal part is timestamp
            PointCloud.pointcloud.push_back(pt);
        }
    }
};


#endif