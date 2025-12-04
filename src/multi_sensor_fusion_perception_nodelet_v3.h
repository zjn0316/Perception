// v3,2025/12/4
#ifndef MULTI_SENSOR_FUSION_PERCEPTION_NODELET_V3_H
#define MULTI_SENSOR_FUSION_PERCEPTION_NODELET_v3_H

#include <nodelet/nodelet.h>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <nlink_parser/LinktrackAoaNodeframe0.h>
#include <people_tracker/PersonArray.h>

#include "yolo_infer.h"          // 你自己的 YOLO 推理接口
#include <cv_bridge/cv_bridge.h>

namespace multi_sensor_fusion{

class MultiSensorFusionPerceptionNodelet : public nodelet::Nodelet{
public:
    MultiSensorFusionPerceptionNodelet();
    virtual ~MultiSensorFusionPerceptionNodelet();
    virtual void onInit() override;

private:
    /*========= ROS 基础设施 =========*/
    ros::NodeHandle nh_, pnh_;

    /*========= 同步器 =========*/
    std::shared_ptr<message_filters::Subscriber<nlink_parser::LinktrackAoaNodeframe0>> sub_uwb_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::CompressedImage>> sub_img_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> sub_depth_;
    std::shared_ptr<message_filters::Subscriber<people_tracker::PersonArray>> sub_leg_;
    
    using ApproxSyncPolicy = message_filters::sync_policies::ApproximateTime<
                       sensor_msgs::CompressedImage,
                       sensor_msgs::Image,
                       nlink_parser::LinktrackAoaNodeframe0,
                       people_tracker::PersonArray>;
    std::shared_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync_;
    
    /*========= 发布器 =========*/
    ros::Publisher pub_marker_;   // 统一发所有 marker

    /*========= TF =========*/
    tf2_ros::Buffer tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

    /*========= 算法内核 =========*/
    std::unique_ptr<YoloDetector> yolo_;

    /*========= 一帧用的临时缓存（非成员！） =========*/
    struct FrameCache{
        ros::Time t;
        /* uwb */
        bool uwb_ok = false;
        geometry_msgs::Point uwb_pt_base;
        double uwb_u = -1, uwb_v = -1;
        /* zed */
        bool zed_ok = false;
        cv_bridge::CvImagePtr cv_ptr;              // 共享_ptr，零拷贝
        cv::Mat depth;                             // share_ptr 零拷贝
        std::vector<Detection> yolo_filt;
        std::vector<cv::Point3f> zed_pts_base;
        /* leg */
        bool leg_ok = false;
        people_tracker::PersonArray leg_msg;
    };

    /*========= 流水线 =========*/
    void spin(const sensor_msgs::CompressedImageConstPtr& img,
              const sensor_msgs::ImageConstPtr& depth,
              const nlink_parser::LinktrackAoaNodeframe0ConstPtr& uwb,
              const people_tracker::PersonArrayConstPtr& leg);

    void uwbWorker(const nlink_parser::LinktrackAoaNodeframe0ConstPtr& msg,
                    FrameCache& c);

    void zedWorker(const sensor_msgs::CompressedImageConstPtr& img,
                    const sensor_msgs::ImageConstPtr& depth,
                    FrameCache& c);

    void legWorker(const people_tracker::PersonArrayConstPtr& leg,
                    FrameCache& c);

    void publishAll(const FrameCache& c);

    /*========= 小工具 =========*/
    inline visualization_msgs::Marker makeSphere(const geometry_msgs::Point& p,
                                                 const std_msgs::ColorRGBA& color,
                                                 const std::string& ns,
                                                 int id) const;
};

} // namespace
#endif