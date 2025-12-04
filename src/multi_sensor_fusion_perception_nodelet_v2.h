// v2,2025/12/4
#ifndef MULTI_SENSOR_FUSION_PERCEPTION_NODELET_V2_H
#define MULTI_SENSOR_FUSION_PERCEPTION_NODELET_V2_H

#include <nodelet/nodelet.h>
#include <ros/ros.h>
#include <geometry_msgs/PointStamped.h>


// messages for visualization
#include <visualization_msgs/Marker.h>// 包含ROS可视化标记消息的头文件，用于在RViz等工具中可视化标记信息
#include <visualization_msgs/MarkerArray.h>// 包含ROS可视化标记数组消息的头文件，用于在RViz等工具中可视化多个标记信息
// tf
#include <tf/transform_listener.h>// 包含TF（坐标变换）库的头文件，用于处理不同坐标系之间的变换
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_eigen/tf2_eigen.h>

// message_filters消息同步器
#include <message_filters/subscriber.h>// 包含消息过滤器订阅者的头文件，用于订阅ROS消息并与消息同步器配合
#include <message_filters/synchronizer.h>// 包含消息同步器的头文件，用于同步来自不同话题的消息
#include <message_filters/sync_policies/approximate_time.h> // 包含近似时间同步策略的头文件，根据时间接近程度同步消息，适用于时间戳不完全一致的情况

// uwb
#include <nlink_parser/LinktrackAoaNodeframe0.h>

// 图像
#include <opencv2/opencv.hpp>// 包含OpenCV库的主头文件，提供图像处理和计算机视觉相关功能
#include <cv_bridge/cv_bridge.h>// 包含cv_bridge库的头文件，用于在ROS图像消息和OpenCV图像之间进行转换
#include <image_transport/image_transport.h>
// yolo11
#include "yolo_utils.h"// 包含自定义的YOLO工具头文件，提供YOLO目标检测相关的工具函数
#include "yolo_infer.h"// 包含自定义的YOLO推理头文件，用于进行YOLO目标检测的推理操作

// 点云
#include <pcl/io/pcd_io.h>// 包含PCL（点云库）的输入输出头文件，用于读写点云数据文件
#include <pcl/point_cloud.h>// 包含PCL点云类的头文件，定义点云的基本数据结构
#include <pcl/point_types.h>// 包含PCL点类型的头文件，定义不同类型的点云点
#include <pcl/filters/passthrough.h> // 包含PCL直通滤波器的头文件，用于过滤点云数据
#include <pcl_conversions/pcl_conversions.h> // 包含PCL与ROS消息转换的头文件，用于在PCL点云和ROS点云消息之间进行转换
#include <sensor_msgs/PointCloud2.h>// 包含ROS点云消息的头文件，用于处理和传输点云数据

// eigen
#include <Eigen/Dense>// 包含Eigen库的稠密矩阵和向量操作头文件，提供高效的线性代数运算功能
#include <Eigen/Geometry>// 包含Eigen库的几何模块头文件，用于处理几何变换，如旋转、平移等

#include <memory>
#include <vector>
#include <unordered_map>
#include <deque>
#include <algorithm>   
#include <iomanip>        // 格式化打印
#include <people_tracker/PersonArray.h>



namespace multi_sensor_fusion {

class MultiSensorFusionPerceptionNodelet : public nodelet::Nodelet {
public:
    MultiSensorFusionPerceptionNodelet();
    virtual ~MultiSensorFusionPerceptionNodelet();

    virtual void onInit() override;

private:
    // ROS成员
    ros::NodeHandle nh;
    ros::NodeHandle pnh;

    // 话题参数成员变量
    std::string caminfo_topic_;
    std::string img_topic_;
    std::string depth_topic_;
    std::string uwb_topic_;
    std::string vis_topic_;
    std::string zed_point_topic_;
    std::string zed_marker_topic_;
    std::string uwb_point_topic_;
    std::string uwb_marker_topic_;
    void loadTopicParameters() ;
    void printTopicConfiguration();

    // tf
    geometry_msgs::TransformStamped tf_base2odom_; // base_link0 to odom_est
    geometry_msgs::TransformStamped tf_odom2base_; // odom_est to base_link0
    geometry_msgs::TransformStamped tf_zed2odom_; // laser to odom_est
    geometry_msgs::TransformStamped tf_zed2base_; // odom_est to laser 
    geometry_msgs::TransformStamped tf_nlink2base_; // nlink to base_link0
    geometry_msgs::TransformStamped tf_nlink2odom_; // nlink to odom_est
    tf2_ros::Buffer tf_buffer_;              // 用于坐标变换
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_; // 必须保持生命周期
    void lookupStaticTransforms();

    // 相机内参
    ros::Subscriber caminfo_sub_;
    cv::Mat transform_i2p; // image->pixel
    cv::Mat transform_c2i; // camera->image
    cv::Mat transform_u2c; // uwb->camera
    cv::Mat transform_u2p; // uwb->pixel
    bool caminfo_updated;
    void caminfoCallback(const sensor_msgs::CameraInfoConstPtr&);

    // uwb
    double uwb_u_ ;
    double uwb_v_ ;
    std::shared_ptr<message_filters::Subscriber<nlink_parser::LinktrackAoaNodeframe0>> sub_uwb_;
    ros::Publisher pub_uwb_;
    ros::Publisher pub_uwb_marker_;
    geometry_msgs::Point uwb_point_odom_; // 存储转换后的UWB位置
    void uwbCallback(const nlink_parser::LinktrackAoaNodeframe0::ConstPtr& uwb_msg);

    // zed
    ros::Publisher pub_yolo_;
    ros::Publisher pub_zed_;
    ros::Publisher pub_zed_marker_;
    std::unique_ptr<YoloDetector> detector_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::CompressedImage>> sub_img_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> sub_depth_;
    void zedCallback(const sensor_msgs::CompressedImageConstPtr& img_msg,
                     const sensor_msgs::ImageConstPtr& depth_msg);
    std::vector<Detection> filterResult(const std::vector<Detection>&);
    float median3x3(const cv::Mat& depth, int u, int v);

    // leg
    // 最新 UWB 和 ZED 坐标（base_link0）
    geometry_msgs::Point latest_uwb_point_base_;
    geometry_msgs::Point latest_zed_point_base_;
    bool has_uwb_point_ = false;
    bool has_zed_point_ = false;
    std::shared_ptr<message_filters::Subscriber<people_tracker::PersonArray>> sub_leg_;
    ros::Publisher  pub_leg_marker_;
    void legCallback(const people_tracker::PersonArray::ConstPtr& leg_msg);
    
    // spin
    using ApproxSyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::CompressedImage,
        sensor_msgs::Image,
        nlink_parser::LinktrackAoaNodeframe0,
        people_tracker::PersonArray>;
    std::shared_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync_;
    void spin(const sensor_msgs::CompressedImageConstPtr& img_msg,
              const sensor_msgs::ImageConstPtr& depth_msg,
              const nlink_parser::LinktrackAoaNodeframe0::ConstPtr& uwb_msg,
              const people_tracker::PersonArray::ConstPtr& leg_msg
              );
  
    void print_transform_matrix();

}; 

} // namespace multi_sensor_fusion

#endif // MULTI_SENSOR_FUSION_PERCEPTION_NODELET_H
