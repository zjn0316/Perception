// v2,2025/12/12
#pragma once

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
#include "deepsort.h"
#include "NvInfer.h"
#include "fusion_net_trt.h" 

struct MaskDepthStats {
    float validRatio;  // 掩码内深度有效像素占比 (0~1)
    float meanDepth;   // 掩码内有效深度均值 (米)
    float varDepth;    // 掩码内有效深度方差 (米²)
};


// 自定义 deepsort Logger 类
class TensorRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // 根据严重级别输出
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                ROS_ERROR("TensorRT INTERNAL_ERROR: %s", msg);
                break;
            case Severity::kERROR:
                ROS_ERROR("TensorRT ERROR: %s", msg);
                break;
            case Severity::kWARNING:
                ROS_WARN("TensorRT WARNING: %s", msg);
                break;
            case Severity::kINFO:
                ROS_INFO("TensorRT INFO: %s", msg);
                break;
            case Severity::kVERBOSE:
                ROS_DEBUG("TensorRT VERBOSE: %s", msg);
                break;
        }
    }
};




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
    std::string fusion_point_topic_;  
    std::string fusion_marker_topic_;
    

    // 话题
    void loadTopicParameters() ;
    void printTopicConfiguration();

    // tf
    geometry_msgs::TransformStamped tf_nlink2cam_;   // nlink → zed2_left_camera_optical_frame
    geometry_msgs::TransformStamped tf_nlink2base_; // nlink → base_link0
    geometry_msgs::TransformStamped tf_nlink2odom_; // nlink → odom_est
    geometry_msgs::TransformStamped tf_base2odom_; // base_link0 → odom_est
    geometry_msgs::TransformStamped tf_odom2base_; // odom_est → base_link0
    geometry_msgs::TransformStamped tf_zed2odom_; // 
    geometry_msgs::TransformStamped tf_zed2base_; // 
    geometry_msgs::TransformStamped tf_base2cam_;// base_link0 → zed2_left_camera_optical_frame

    tf2_ros::Buffer tf_buffer_;              // 用于坐标变换
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_; // 必须保持生命周期
    void lookupStaticTransforms();

    // 相机内参
    ros::Subscriber caminfo_sub_;
    cv::Mat P; // 投影矩阵 P
    cv::Mat K; // 从 P 矩阵提取前 3x3 部分作为内参矩阵 K
    bool caminfo_updated;
    void caminfoCallback(const sensor_msgs::CameraInfoConstPtr&);

    // uwb
    float uwb_conf_;
    cv::Point2d uwb_uv_;
    std::shared_ptr<message_filters::Subscriber<nlink_parser::LinktrackAoaNodeframe0>> sub_uwb_;
    ros::Publisher pub_uwb_;
    ros::Publisher pub_uwb_marker_;
    geometry_msgs::PointStamped target_uwb_;
    // 计算 UWB 置信度
    float calcUwbConfidence(float fp_rssi, float rx_rssi, float dis) const;
    cv::Point2d camera2pixel(const geometry_msgs::PointStamped& pt_cam,const cv::Matx33d& K) ;
    void printUwbPositions(const geometry_msgs::PointStamped& uwb_base,
                           const geometry_msgs::PointStamped& uwb_odom,
                           const geometry_msgs::PointStamped& uwb_cam,
                           const cv::Point2d& uwb_uv);

    // leg
    float leg_conf_;
    cv::Point2d leg_uv_;
    std::shared_ptr<message_filters::Subscriber<people_tracker::PersonArray>> sub_leg_;
    ros::Publisher pub_leg_;
    ros::Publisher  pub_leg_marker_;
    geometry_msgs::PointStamped target_leg_;
    float best_leg_d2_ = 0.5f;
    int best_leg_idx_ = -1;
    void printLegPositions(const geometry_msgs::PointStamped& leg_odom,
                           const geometry_msgs::PointStamped& leg_base,
                           const geometry_msgs::PointStamped& leg_cam,
                           const cv::Point2d& leg_uv);

    // zed
    float zed_conf_;
    cv::Point2d zed_uv_;
    ros::Publisher pub_yolo_;
    ros::Publisher pub_zed_;
    ros::Publisher pub_zed_marker_;
    double yolo_conf_threshold_;
    std::string yolo_engine_path_; 
    std::unique_ptr<YoloDetector> yolo_;
    int deepsort_max_age_, deepsort_n_init_;
    double deepsort_max_iou_, deepsort_max_cos_;
    std::string deepsort_engine_path_; 
    std::unique_ptr<DeepSort> deepsort_;
    geometry_msgs::PointStamped target_zed_;
    float best_zed_d2_ = 0.5f;
    int best_zed_idx_ = -1;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::CompressedImage>> sub_img_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> sub_depth_;
    std::vector<Detection> filterResult(const std::vector<Detection>&);
    std::vector<DetectBox> toDetectBoxPed(const std::vector<Detection>& src);
    float median3x3(const cv::Mat& depth, int u, int v);
    cv::Vec3f pixel2camera(const cv::Point2d& uv,float z,const cv::Matx33d& K);
    float depthInsideMask(const std::vector<float>& maskMatrix,
                          const cv::Mat& depthMap,
                          const cv::Rect& bbox,
                          float thr = 0.5f);
    MaskDepthStats maskDepthStats(const std::vector<float>& maskMatrix,
                              const cv::Mat& depthMap,
                              const cv::Rect& bbox,
                              float thr = 0.5f);
    float calcZedConfidence(float yolo_conf,
                           float valid_ratio,
                           float depth_var);
    void printZedPositions(const geometry_msgs::PointStamped& zed_cam,
                    const geometry_msgs::PointStamped& zed_base,
                    const geometry_msgs::PointStamped& zed_odom,
                    const cv::Point2d&                 zed_uv,
                    int                                track_id);

    // fusion
    std::string fusion_engine_path_; 
    std::unique_ptr<fusion_net::FusionNetTRT> fusion_;
    std::vector<float> seq_buf_;   
    ros::Publisher pub_fusion_;
    ros::Publisher pub_fusion_marker_;
    // 辅助函数
    void pushSensorToSeq(const geometry_msgs::PointStamped& uwb,
                       const geometry_msgs::PointStamped& leg,
                       const geometry_msgs::PointStamped& zed,
                       float uwb_conf, float leg_conf, float zed_conf);
    void doFusionInferAndPub();   // 推理+发布结果
    
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
    void print_confidences(float uwb_conf, float zed_conf, float leg_conf);
    void print_positions(const geometry_msgs::PointStamped& uwb_pt_base,
                         const geometry_msgs::PointStamped& zed_pt_base,
                         const geometry_msgs::PointStamped& leg_pt_base);
    void print_transform_matrix(const std::string& from_frame,const std::string& to_frame);

}; 

} // namespace multi_sensor_fusion


