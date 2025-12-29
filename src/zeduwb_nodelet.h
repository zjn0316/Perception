#ifndef ZED_UWB_NODELET_H
#define ZED_UWB_NODELET_H

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

// 1. 消息过滤器与同步
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// 2. 消息类型
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <geometry_msgs/PointStamped.h>
#include <std_msgs/Float32MultiArray.h> // 用于发布 bbox
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

// 3. 图像处理与 TF
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.h>

// 4. UWB 消息
#include <nlink_parser/LinktrackAoaNodeframe0.h>

// 5. YOLO 头文件 (假设您的工程中有这个头文件，定义了 YoloDetector 类和 Detection 结构体)
#include "infer.h"   // 对应上传的 infer.h
#include "types.h"   // 对应上传的 types.h



namespace zeduwb_ns {

class ZedUwbNodelet : public nodelet::Nodelet {
public:
    virtual void onInit();

private:
    // 同步回调函数：同时接收 RGB图、深度图、UWB点
    void syncCallback(const sensor_msgs::ImageConstPtr& img_msg,
                      const sensor_msgs::ImageConstPtr& depth_msg,
                      const nlink_parser::LinktrackAoaNodeframe0ConstPtr& uwb_msg);

    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;

    // --- 消息同步定义 ---
    // 定义同步策略：ApproximateTime (近似时间同步)，因为 UWB 和 相机频率不同
    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::Image, 
        sensor_msgs::Image, 
        nlink_parser::LinktrackAoaNodeframe0
    > ApproxSyncPolicy;

    // 订阅者 (必须用 shared_ptr 管理，否则会报错)
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> img_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> depth_sub_;
    std::shared_ptr<message_filters::Subscriber<nlink_parser::LinktrackAoaNodeframe0>> uwb_sub_;
    
    // 同步器
    std::shared_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync_;

    // --- 发布者 ---
    ros::Publisher pub_image_compressed_; // 对应 self.topic_img (CompressedImage)
    ros::Publisher pub_uwb_3d_;           // 对应 self.topic_uwb_3d (PointStamped)
    ros::Publisher pub_uwb_2d_;           // 对应 self.topic_uwb_2d (PointStamped)
    ros::Publisher pub_bbox_;             // 对应 self.topic_bbox (Float32MultiArray)
    ros::Publisher pub_uwb_marker_;       // 对应 self.topic_uwb_marker (Marker)

    // --- 坐标变换 (TF) ---
    tf2_ros::Buffer tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // 算法成员
    cv::Mat K_;
    std::unique_ptr<YoloDetector> yolo_; // [新增] YOLO 指针

    // 参数
    std::string yolo_engine_path_;

    // 辅助函数
    std::vector<Detection> FilterResult(const std::vector<Detection>& in);
};

} // namespace zeduwb_ns

#endif