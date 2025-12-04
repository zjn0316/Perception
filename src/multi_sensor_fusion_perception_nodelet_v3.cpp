// v3,2025/12/4
#include "multi_sensor_fusion_perception_nodelet_v3.h"
#include <pluginlib/class_list_macros.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <opencv2/imgproc.hpp>

namespace multi_sensor_fusion{

/*==================== 生命周期 ====================*/
MultiSensorFusionPerceptionNodelet::MultiSensorFusionPerceptionNodelet(){}
MultiSensorFusionPerceptionNodelet::~MultiSensorFusionPerceptionNodelet(){}

void MultiSensorFusionPerceptionNodelet::onInit(){
    nh_ = getNodeHandle();
    pnh_ = getPrivateNodeHandle();

    /* 参数可动态加载，这里用固定话题 */
    std::string yolo_engine;
    pnh_.param<std::string>("yolo_engine", yolo_engine,
        "/home/zjn/catkin_ws/src/multi_sensor_fusion_perception_nodelet/engine/yolo11s-seg.plan");
    yolo_.reset(new YoloDetector(yolo_engine));// 创建 YOLO 推理器
    
    /* tf */
    tf_listener_.reset(new tf2_ros::TransformListener(tf_buffer_));

    /* 订阅 */
    sub_img_.reset(new message_filters::Subscriber<sensor_msgs::CompressedImage>(nh_,"/zed2/zed_node/rgb/image_rect_color/compressed", 1));
    sub_depth_.reset(new message_filters::Subscriber<sensor_msgs::Image>(nh_,"/zed2/zed_node/depth/depth_registered", 1)); 
    sub_uwb_.reset(new message_filters::Subscriber<nlink_parser::LinktrackAoaNodeframe0>(nh_, "/uwb_filter_polar", 10));
    sub_leg_.reset(new message_filters::Subscriber<people_tracker::PersonArray>(nh_, "/people_tracked", 2));
    /* 同步 */
    sync_.reset(new message_filters::Synchronizer<ApproxSyncPolicy>(ApproxSyncPolicy(50), *sub_img_, *sub_depth_, *sub_uwb_, *sub_leg_));
    sync_->registerCallback(boost::bind(&MultiSensorFusionPerceptionNodelet::spin, this, _1, _2, _3, _4));

    /* 发布 */
    pub_marker_ = nh_.advertise<visualization_msgs::MarkerArray>("/fusion_marker", 1);

    NODELET_INFO("Multi-sensor fusion nodelet ready.");
}

/*==================== 主流水线 ====================*/
void MultiSensorFusionPerceptionNodelet::spin(
      const sensor_msgs::CompressedImageConstPtr& img,
      const sensor_msgs::ImageConstPtr& depth,
      const nlink_parser::LinktrackAoaNodeframe0ConstPtr& uwb,
      const people_tracker::PersonArrayConstPtr& leg)
{
    NODELET_INFO("spin ready.");
    FrameCache c;          // ① 单帧缓存，栈上
    c.t = img->header.stamp;

    /* ② 顺序执行，后边直接用前面结果 */
    uwbWorker(uwb, c);
    zedWorker(img, depth, c);
    legWorker(leg, c);

    /* ③ 一次性发布 */
    publishAll(c);
    NODELET_INFO("spin done.");
}

/* ============================================================
 * UWB  Worker：极坐标 → base_link0
 * ============================================================ */
void MultiSensorFusionPerceptionNodelet::uwbWorker(
    const nlink_parser::LinktrackAoaNodeframe0ConstPtr& msg,
    FrameCache& c)
{
    if (msg->nodes.empty() || msg->nodes[0].role != 1) return;

    /* 极坐标 → nlink 直角 → base_link0 */
    float d = msg->nodes[0].dis;                 // 距离
    float a = msg->nodes[0].angle * M_PI / 180.f; // 角度转弧度
    geometry_msgs::PointStamped nlink, base;     // 两个临时变量
    nlink.header = msg->header;                  // 时间戳复用
    nlink.point.x = d * cos(a);                  // 极坐标 → 直角
    nlink.point.y = d * sin(a);
    nlink.point.z = 0.5;                         // 固定高度

    try{
        /* TF：nlink → base_link0 */
        geometry_msgs::TransformStamped tf =tf_buffer_.lookupTransform("base_link0", nlink.header.frame_id,
                                    ros::Time(0), ros::Duration(0.02));
        tf2::doTransform(nlink, base, tf);// 执行变换
    }catch (tf2::TransformException& ex){
        ROS_WARN_THROTTLE(1.0, "uwb tf: %s", ex.what());
        return;
    }

    c.uwb_pt_base = base.point;// 写进缓存
    c.uwb_ok = true;           // 打标志
    NODELET_INFO("uwbWorker has done.");
}

/*==================== zed ====================*/
void MultiSensorFusionPerceptionNodelet::zedWorker(
    const sensor_msgs::CompressedImageConstPtr& img,
    const sensor_msgs::ImageConstPtr& depth,
    FrameCache& c)
{
    /* 1. 零拷贝取图 */
    c.cv_ptr = cv_bridge::toCvCopy(img, "bgr8");// 把压缩图像解码成 cv::Mat，颜色空间 BGR8
    c.depth  = cv_bridge::toCvShare(depth, sensor_msgs::image_encodings::TYPE_32FC1)->image;// 共享方式把 ROS 深度图转成 32FC1 的 cv::Mat

    /* 2. YOLO */
    c.yolo_filt = yolo_->inference(c.cv_ptr->image);// 对当前帧做推理，返回过滤后的检测框

    /* 3. 用 uwb 像素门控（如果 uwb 已算完） */
    if (c.uwb_ok){
        const double fx = 267.4111022949219, fy = 267.4111022949219;
        const double cx = 312.1522827148438, cy = 184.7516326904297;

        geometry_msgs::PointStamped cam_pt;
        cam_pt.header.frame_id = "zed2_left_camera_optical_frame";
        cam_pt.point = c.uwb_pt_base;
        try{
        geometry_msgs::TransformStamped tf =
            tf_buffer_.lookupTransform("zed2_left_camera_optical_frame",
                                        "base_link0", ros::Time(0), ros::Duration(0.02));
        tf2::doTransform(cam_pt, cam_pt, tf);
        }catch(...){ c.uwb_ok = false; goto no_gate; }

        // 把 3D 点投影到像素坐标
        double Z = cam_pt.point.z;
        c.uwb_u = fx * cam_pt.point.x / Z + cx;
        c.uwb_v = fy * cam_pt.point.y / Z + cy;

        /* 简单像素门控：只保留检测框中心与 UWB 投影距离 < 80 pix 的目标 */
        c.yolo_filt.erase(
        std::remove_if(c.yolo_filt.begin(), c.yolo_filt.end(),
            [&](const Detection& d){
                int cx = (d.bbox[0] + d.bbox[2]) * 0.5f;
                int cy = (d.bbox[1] + d.bbox[3]) * 0.5f;
                return std::hypot(cx - c.uwb_u, cy - c.uwb_v) > 80;
            }), c.yolo_filt.end());
    }
no_gate:// 门控失败或关闭时跳到这里

    /* 4. 深度 → 3-D → base_link0 */
    for (const auto& d : c.yolo_filt){
        // 只保留 "人" 类别
        if (d.classId != 0) continue;
        // 计算检测框中心像素坐标
        int u = (d.bbox[0] + d.bbox[2]) * 0.5f;
        int v = (d.bbox[1] + d.bbox[3]) * 0.5f;
        // 防越界
        u = (u < 0 ? 0 : (u >= c.depth.cols ? c.depth.cols - 1 : u));
        v = (v < 0 ? 0 : (v >= c.depth.rows ? c.depth.rows - 1 : v));
        float z = c.depth.at<float>(v, u) * 0.001f;   // mm → m
        if (z <= 0.f || z > 10.f) continue;
        // 反投影到相机坐标系
        const double fx = 267.4111022949219, fy = 267.4111022949219;
        const double cx = 312.1522827148438, cy = 184.7516326904297;
        cv::Point3f cam((u - cx) / fx * z, (v - cy) / fy * z, z);

        try{
            // 把相机坐标点通过 TF 转到 base_link0
            geometry_msgs::Vector3Stamped in, out;
            in.vector.x = cam.x; in.vector.y = cam.y; in.vector.z = cam.z;
            in.header.frame_id = "zed2_left_camera_optical_frame";
            geometry_msgs::TransformStamped tf =
                tf_buffer_.lookupTransform("base_link0", in.header.frame_id,
                                            ros::Time(0), ros::Duration(0.02));
            tf2::doTransform(in, out, tf);
            c.zed_pts_base.emplace_back(out.vector.x, out.vector.y, out.vector.z);
        }catch(...){}
    }
    c.zed_ok = true;
    NODELET_INFO("zedWorker has done.");
}

/*==================== leg ====================*/
void MultiSensorFusionPerceptionNodelet::legWorker(
      const people_tracker::PersonArrayConstPtr& msg,
      FrameCache& c)
{
    c.leg_msg = *msg;   // 轻量拷贝

    /* 用 uwb/zed 做最近邻关联 */
    if (c.uwb_ok){
        // 遍历腿检测输出，若与 UWB 水平距离 < 0.5 m 则强制 z=0.3 m（贴地）
        for (auto& p : c.leg_msg.people){
        double dx = p.pose.position.x - c.uwb_pt_base.x;
        double dy = p.pose.position.y - c.uwb_pt_base.y;
        if (std::hypot(dx, dy) < 0.5){
            p.pose.position.z = 0.3;   // 强制贴地
        }
        }
    }
    c.leg_ok = true;
    NODELET_INFO("legWorker has done.");
}

/*==================== 发布 ====================*/
void MultiSensorFusionPerceptionNodelet::publishAll(const FrameCache& c){
    visualization_msgs::MarkerArray ma;
    int id = 0;
    // lambda 快速生成颜色
    auto make_color = [](float r, float g, float b){
        std_msgs::ColorRGBA c; c.r = r; c.g = g; c.b = b; c.a = 1; return c;};
    
    // UWB 结果：蓝色球
    if (c.uwb_ok)
        ma.markers.push_back(makeSphere(c.uwb_pt_base, make_color(0, 0, 1), "uwb", id++));
    
    // ZED 结果：红色球
    for (const auto& pt : c.zed_pts_base){//红色
        geometry_msgs::Point gp;   // ① 在循环里就地声明
        gp.x = pt.x;
        gp.y = pt.y;
        gp.z = pt.z;
        ma.markers.push_back(makeSphere(gp, make_color(1, 0, 0), "zed", id++));
    }
    // leg 结果：绿色球
    for (const auto& p : c.leg_msg.people)//绿色
        ma.markers.push_back(makeSphere(p.pose.position, make_color(0, 1, 0), "leg", id++));
    pub_marker_.publish(ma);
    NODELET_INFO("publishAll has done.");
}

/*==================== 工具 ====================*/
// 快速生成一个球体 marker 的辅助函数
inline visualization_msgs::Marker
MultiSensorFusionPerceptionNodelet::makeSphere(const geometry_msgs::Point& p,
                                               const std_msgs::ColorRGBA& color,
                                               const std::string& ns,
                                               int id) const
{
    visualization_msgs::Marker m;
    m.header.frame_id = "base_link0";// 固定坐标系
    m.header.stamp    = ros::Time::now();
    m.ns = ns;  
    m.id = id;
    m.type = m.SPHERE;  // 球体
    m.action = m.ADD;
    m.pose.position = p;
    m.pose.orientation.w = 1;
    m.scale.x = m.scale.y = m.scale.z = 0.2;// 直径 0.2 m
    m.color = color;  // 颜色
    m.lifetime = ros::Duration(0.2);
    return m;
}

} // namespace

PLUGINLIB_EXPORT_CLASS(multi_sensor_fusion::MultiSensorFusionPerceptionNodelet, nodelet::Nodelet)