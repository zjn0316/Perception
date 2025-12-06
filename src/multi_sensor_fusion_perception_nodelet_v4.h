// v2,2025/12/4
#ifndef MULTI_SENSOR_FUSION_PERCEPTION_NODELET_V4_H
#define MULTI_SENSOR_FUSION_PERCEPTION_NODELET_V4_H

// 1. 系统头文件
// #include <algorithm>
// #include <deque>
// #include <vector>
// #include <memory>

// 2. 第三方库头文件
// ROS 基础
#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <geometry_msgs/PointStamped.h>
#include <image_transport/image_transport.h>
// tf
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>
// 可视化Marker
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
// message_filters
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
// nlink
#include <nlink_parser/LinktrackAoaNodeframe0.h>
// people_tracker
#include <people_tracker/PersonArray.h>
// OpenCV
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>

// 3. 项目自身头文件
#include "yolo_infer.h"




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
    // 订阅
    std::string uwb_topic_;
    std::string caminfo_topic_;
    std::string img_topic_;
    std::string depth_topic_;
    std::string leg_topic_;  
    // 发布
    std::string uwb_point_topic_;
    std::string uwb_marker_topic_; 
    std::string zed_point_topic_;
    std::string zed_marker_topic_;
    std::string vis_topic_;
    std::string leg_point_topic_;         
    std::string leg_marker_topic_;

    void loadTopicParameters() ;
    void printTopicConfiguration();

    // tf
    geometry_msgs::TransformStamped tf_nlink2cam_;   // nlink → zed2_left_camera_optical_frame
    geometry_msgs::TransformStamped tf_nlink2base_; // nlink → base_link0
    geometry_msgs::TransformStamped tf_nlink2odom_; // nlink → odom_est
    geometry_msgs::TransformStamped tf_base2odom_; // base_link0 → odom_est
    geometry_msgs::TransformStamped tf_odom2base_; // odom_est → base_link0
    geometry_msgs::TransformStamped tf_zed2odom_; // laser → odom_est
    geometry_msgs::TransformStamped tf_zed2base_; // odom_est → laser 

    tf2_ros::Buffer tf_buffer_;              // 用于坐标变换
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_; // 必须保持生命周期
    void lookupStaticTransforms();

    // 相机内参
    ros::Subscriber caminfo_sub_;
    cv::Mat P; // 投影矩阵 P
    cv::Mat K; // 从 P 矩阵提取前 3x3 部分作为内参矩阵 K
    cv::Mat transform_c2i; // camera->image
    cv::Mat transform_u2c; // uwb->camera
    cv::Mat transform_u2p; // uwb->pixel
    bool caminfo_updated;
    void caminfoCallback(const sensor_msgs::CameraInfoConstPtr&);

    // uwb
    cv::Point2d uwb_uv_;
    std::shared_ptr<message_filters::Subscriber<nlink_parser::LinktrackAoaNodeframe0>> sub_uwb_;
    ros::Publisher pub_uwb_;
    ros::Publisher pub_uwb_marker_;
    cv::Point2d camera2pixel(const geometry_msgs::PointStamped& pt_cam,const cv::Matx33d& K) ;

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
    cv::Vec3f pixel2camera(const cv::Point2d& uv,float z,const cv::Matx33d& K);

    // leg
    geometry_msgs::Point latest_uwb_point_base_;
    geometry_msgs::Point latest_zed_point_base_;
    bool has_uwb_point_ = false;
    bool has_zed_point_ = false;
    std::shared_ptr<message_filters::Subscriber<people_tracker::PersonArray>> sub_leg_;
    ros::Publisher pub_leg_;
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
    // 工具函数
    void print_transform_matrix(const std::string& from_frame,const std::string& to_frame);

}; 

} // namespace multi_sensor_fusion

#endif // MULTI_SENSOR_FUSION_PERCEPTION_NODELET_v4_H
