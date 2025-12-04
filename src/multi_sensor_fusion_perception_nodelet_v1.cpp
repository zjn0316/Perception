// v1,2025/11/22
#include "multi_sensor_fusion_perception_nodelet.h"
#include <pluginlib/class_list_macros.h>

namespace multi_sensor_fusion {

MultiSensorFusionPerceptionNodelet::MultiSensorFusionPerceptionNodelet():caminfo_updated(false){}

MultiSensorFusionPerceptionNodelet::~MultiSensorFusionPerceptionNodelet() {}

void MultiSensorFusionPerceptionNodelet::onInit() {
    setlocale(LC_ALL,"");
    NODELET_INFO("正在初始化 Multi-Sensor Fusion Perception Nodelet...");
    auto start = ros::Time::now();

    nh = getNodeHandle();
    pnh = getPrivateNodeHandle();

    loadTopicParameters();
    printTopicConfiguration();
    
    // tf
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(tf_buffer_);

    // caminfo
    caminfo_sub_ = nh.subscribe("/zed2/zed_node/rgb/camera_info", 10, &MultiSensorFusionPerceptionNodelet::caminfoCallback, this);

    // uwb
    sub_uwb_.reset(new message_filters::Subscriber<nlink_parser::LinktrackAoaNodeframe0>(nh, "/uwb_filter_polar", 10));
    pub_uwb_ = nh.advertise<geometry_msgs::PointStamped>("/uwb_point", 1);
    pub_uwb_marker_ = nh.advertise<visualization_msgs::Marker>("/uwb_point_marker", 1);

    // zed
    detector_.reset(new YoloDetector("/home/zjn/catkin_ws/src/multi_sensor_fusion_perception_nodelet/engine/yolo11s-seg.plan"));
    sub_img_.reset(new message_filters::Subscriber<sensor_msgs::CompressedImage>(nh,"/zed2/zed_node/rgb/image_rect_color/compressed", 1));
    sub_depth_.reset(new message_filters::Subscriber<sensor_msgs::Image>(nh,"/zed2/zed_node/depth/depth_registered", 1)); 
    pub_yolo_ = nh.advertise<sensor_msgs::Image>("/yolo_imgae_seg", 1);
    pub_zed_ = nh.advertise<geometry_msgs::PointStamped>("/zed_point", 1);
    pub_zed_marker_ = nh.advertise<visualization_msgs::MarkerArray>("/zed_point_marker", 1);

    // leg
    sub_leg_.reset(new message_filters::Subscriber<people_tracker::LegArray>(nh, "/dr_spaam_detection_array", 2));
    pub_leg_marker_ = nh.advertise<visualization_msgs::MarkerArray>("/leg_point_marker", 1);

    // spin
    sync_.reset(new message_filters::Synchronizer<ApproxSyncPolicy>(ApproxSyncPolicy(50), *sub_img_, *sub_depth_, *sub_uwb_, *sub_leg_));
    sync_->registerCallback(boost::bind(&MultiSensorFusionPerceptionNodelet::spin, this, _1, _2, _3,_4));

    // 跟踪器
    initializeTracker();
    pub_tracked_person_ = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/tracked_person", 1);
    pub_tracking_status_ = nh.advertise<std_msgs::String>("/tracking_status", 1);
    pub_person_velocity_ = nh.advertise<geometry_msgs::Vector3Stamped>("/person_velocity", 1);

    auto duration = (ros::Time::now() - start).toSec();
    ROS_INFO_STREAM("onInit 花费 " << duration * 1000 << " ms");
    NODELET_INFO("Multi-Sensor Fusion Perception Nodelet 已成功初始化！");
}


void MultiSensorFusionPerceptionNodelet::uwbCallback(const nlink_parser::LinktrackAoaNodeframe0::ConstPtr& uwb_msg)
{
    /* -------------- 1. 有效性检查 -------------- */
    if (uwb_msg->nodes.empty() || uwb_msg->nodes.at(0).role != 1) return;

    /* -------------- 2. 极坐标 → nlink 直角坐标 -------------- */
    float dis = uwb_msg->nodes.at(0).dis;
    float ang = uwb_msg->nodes.at(0).angle * M_PI / 180.0f;  // °→rad
    geometry_msgs::PointStamped uwb_pt_nlink;
    uwb_pt_nlink.header.stamp    = uwb_msg->header.stamp;
    uwb_pt_nlink.header.frame_id = "nlink";
    uwb_pt_nlink.point.x = dis * std::cos(ang);
    uwb_pt_nlink.point.y = dis * std::sin(ang);
    uwb_pt_nlink.point.z = 0.5;  // uwb坐标系下固定高度 0.23 m

    /* -------------- 3. uwb坐标投影到像素平面 -------------- */
    // nlink → 图像坐标系
    geometry_msgs::PointStamped uwb_pt_cam;
    try 
    {
        tf_buffer_.transform(uwb_pt_nlink, uwb_pt_cam,"zed2_left_camera_optical_frame",ros::Duration(0.05));   // 50 ms 容忍
    } 
    catch (tf2::TransformException& ex) 
    {
        ROS_WARN_THROTTLE(1.0, "UWB TF fail: %s", ex.what());
        return;
    }

    const double fx = 267.4111022949219;
    const double fy = 267.4111022949219;
    const double cx = 312.1522827148438;
    const double cy = 184.7516326904297;

    double X = uwb_pt_cam.point.x;
    double Y = uwb_pt_cam.point.y;
    double Z = uwb_pt_cam.point.z;

    // 图像 → 像素坐标系
    uwb_u_ = fx * X / Z + cx;
    uwb_v_ = fy * Y / Z + cy;

    /* -------------- 4. 坐标变换：nlink → base_link0 → odom_est -------------- */
    geometry_msgs::PointStamped uwb_pt_base;
    tf2::doTransform(uwb_pt_nlink, uwb_pt_base, tf_nlink2base_);  // nlink → base
    // ROS_INFO("UWB in odom_est: x=%.2f, y=%.2f, z=%.2f",uwb_pt_odom.point.x, uwb_pt_odom.point.y, uwb_pt_odom.point.z);
    geometry_msgs::PointStamped uwb_pt_odom;
    tf2::doTransform(uwb_pt_base, uwb_pt_odom, tf_base2odom_);  // base → odom
    ROS_INFO("UWB in base_link0: x=%.2f, y=%.2f, z=%.2f",uwb_pt_base.point.x, uwb_pt_base.point.y, uwb_pt_base.point.z);
    latest_uwb_point_base_ = uwb_pt_base.point;
    has_uwb_point_ = true;
    
    /*-------------- 5 RViz 可视化：发布 Marker --------------*/
    visualization_msgs::Marker marker;
    marker.header = uwb_pt_base.header;//uwb_pt_odom/uwb_pt_base
    marker.ns = "uwb_detection";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position = uwb_pt_base.point;//uwb_pt_odom/uwb_pt_base
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.2;
    marker.scale.y = 0.2;
    marker.scale.z = 0.2;
    marker.color.r = 0.0f;
    marker.color.g = 0.0f;
    marker.color.b = 1.0f;
    marker.color.a = 1.0f;
    marker.lifetime = ros::Duration(1); // 持续 0.5 秒

    pub_uwb_marker_.publish(marker);
}


void MultiSensorFusionPerceptionNodelet::zedCallback(
    const sensor_msgs::CompressedImageConstPtr& img_msg,
    const sensor_msgs::ImageConstPtr& depth_msg)
{
    /*----------------------------------------------------
     *  1. 解压彩色图 2ms
     *---------------------------------------------------*/
    cv_bridge::CvImagePtr cv_ptr;
    try 
    {
        cv_ptr = cv_bridge::toCvCopy(img_msg, "bgr8");
    } 
    catch (cv_bridge::Exception& e) 
    {
        NODELET_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    cv::Mat img = cv_ptr->image;

    /*----------------------------------------------------
     *  2. YOLO 检测 + 过滤 10ms
     *---------------------------------------------------*/
    std::vector<Detection> res    = detector_->inference(img);
    std::vector<Detection> res_fd = filterResult(res);

    /*----------------------------------------------------
     *  3. 深度图 → cv::Mat (32FC1) 0ms
     *---------------------------------------------------*/
    cv::Mat depth;
    try 
    {
        depth = cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1)->image;
    } 
    catch (cv_bridge::Exception& e) 
    {
        NODELET_ERROR("depth cv_bridge exception: %s", e.what());
        return;
    }

    /*----------------------------------------------------
     *  4. 像素坐标系投影到图像坐标系 2-D 框中心 → 3-D 坐标
     *---------------------------------------------------*/
    // 真实内参 640×360 分辨率
    const double fx = 267.4111022949219;
    const double fy = 267.4111022949219;
    const double cx = 312.1522827148438;
    const double cy = 184.7516326904297;

    visualization_msgs::MarkerArray marker_array;
    int marker_id = 0;  // 每帧从 0 开始编号即可，RViz 靠 namespace+id 区分

    for (const auto& d : res_fd) {
        if (d.classId != 0) continue;  // 只处理行人
        /* 4.1 计算图像中心 */
        int zed_u = static_cast<int>((d.bbox[0] + d.bbox[2]) * 0.5f);
        int zed_v = static_cast<int>((d.bbox[1] + d.bbox[3]) * 0.5f);
        zed_u = std::max(0, std::min(zed_u, depth.cols - 1));
        zed_v = std::max(0, std::min(zed_v, depth.rows - 1));
        /* 4.2 根据像素索引得到深度值*/
        float z_raw = median3x3(depth, zed_u, zed_v);
        if (z_raw >= 15000 || z_raw <= 0.f) continue;
        float z_cam = z_raw * 0.001f;
        /* 4.3 像素 → 图像坐标系 */
        double x_cam = (zed_u - cx) / fx ;
        double y_cam = (zed_v - cy) / fy ;
        /* 4.4 构造相机坐标系下坐标 */
        geometry_msgs::PointStamped zed_pt_cam;
        zed_pt_cam.header.stamp    = depth_msg->header.stamp;
        zed_pt_cam.header.frame_id = "zed2_left_camera_optical_frame";
        zed_pt_cam.point.x         = x_cam;
        zed_pt_cam.point.y         = y_cam;
        zed_pt_cam.point.z         = z_cam;
        // 4.5 odom_est下行人坐标
        geometry_msgs::PointStamped zed_pt_odom;
        zed_pt_odom.header.stamp    = zed_pt_cam.header.stamp; 
        tf2::doTransform(zed_pt_cam, zed_pt_odom, tf_zed2odom_);  // 用成员变量
        // 4.6 base_link0下行人坐标
        geometry_msgs::PointStamped zed_pt_base;
        zed_pt_base.header.stamp    = zed_pt_cam.header.stamp; 
        tf2::doTransform(zed_pt_odom, zed_pt_base, tf_odom2base_);  // 用成员变量
        ROS_INFO("ZED in base_link0: x=%.2f, y=%.2f, z=%.2f",zed_pt_base.point.x, zed_pt_base.point.y, zed_pt_base.point.z);
        latest_zed_point_base_ = zed_pt_base.point;
        has_zed_point_ = true;
        // 4.7 发布 marker
        visualization_msgs::Marker marker;
        marker.header                = zed_pt_base.header;
        marker.ns                    = "zed_detection";
        marker.id                    = marker_id++;  // 关键：保证同一帧内不重复
        marker.type                  = visualization_msgs::Marker::SPHERE;
        marker.action                = visualization_msgs::Marker::ADD;
        marker.pose.position         = zed_pt_base.point;
        marker.pose.orientation.w    = 1.0;
        marker.scale.x = marker.scale.y = marker.scale.z = 0.2;
        marker.color.r = 1.0f;
        marker.color.g = 0.0f;
        marker.color.b = 0.0f;
        marker.color.a = 1.0f;
        marker.lifetime = ros::Duration(1.0);

        marker_array.markers.push_back(marker);
    }
    pub_zed_marker_.publish(marker_array); 

    /*----------------------------------------------------
     *  5. 可视化：带框图
     *---------------------------------------------------*/
    bool found = !res_fd.empty();// 找到行人
    cv::Mat& img_seg = cv_ptr->image;   // 不克隆
    // cv::Mat img_seg = img.clone();      // 克隆，普通对象
    // 画框图
    YoloDetector::draw_image(img_seg, res_fd);
    // 画 zed 红点
    for (const auto& d : res_fd) {
        if (d.classId != 0) continue;
        int zed_u = static_cast<int>((d.bbox[0] + d.bbox[2]) * 0.5f);
        int zed_v = static_cast<int>((d.bbox[1] + d.bbox[3]) * 0.5f);
        cv::circle(img_seg, cv::Point(zed_u, zed_v), 4, cv::Scalar(0, 0, 255), -1);
    }
    // 画 UWB 蓝点
    if (uwb_u_ > 0 && uwb_v_ > 0) {
        cv::circle(img_seg, cv::Point(static_cast<int>(uwb_u_), static_cast<int>(uwb_v_)), 6, cv::Scalar(255, 0, 0), -1);
    }
    sensor_msgs::ImagePtr msg_seg = cv_bridge::CvImage(img_msg->header, "bgr8", img_seg).toImageMsg();
    pub_yolo_.publish(msg_seg);          // 对应 /yolo_imgae_seg"


    /*---------------------------------------------------*/
    last_zed_detections_ = res_fd;                 // 整包保存
    /* 找到距离 UWB 最近的那个行人作为“要跟随的目标” */
    double min_d2 = std::numeric_limits<double>::max();
    for (const auto& d : res_fd) {
        if (d.classId != 0) continue;
        int u = (d.bbox[0] + d.bbox[2]) * 0.5f;
        int v = (d.bbox[1] + d.bbox[3]) * 0.5f;
        float z = median3x3(depth, u, v) * 0.001f;
        double dx = u - uwb_u_;          // 与 UWB 投影点的像素差
        double dy = v - uwb_v_;
        double d2 = dx*dx + dy*dy;
        if (d2 < min_d2) {               // 选最近似 UWB 的那个框
            min_d2 = d2;
            last_zed_bbox_  = cv::Rect2d(d.bbox[0], d.bbox[1],
                                        d.bbox[2]-d.bbox[0],
                                        d.bbox[3]-d.bbox[1]);
            last_zed_depth_ = z;
        }
    }
    depth_ = depth;   // 成员变量赋值，给遮挡检测用
}

std::vector<Detection> MultiSensorFusionPerceptionNodelet::filterResult(const std::vector<Detection>& in){
    std::vector<Detection> out;
    for(const auto& t : in){
        if(t.classId != 0) continue;
        if (t.conf < 0.65f) continue;      // 只保留置信度 > 65%
        float w = t.bbox[2] - t.bbox[0];
        float h = t.bbox[3] - t.bbox[1];
        float r = h / w;
        if(r > 1.2f && r < 4.0f) out.push_back(t);
    }
    return out;
}

// 返回 3×3 邻域中值，边界自动镜像
float MultiSensorFusionPerceptionNodelet::median3x3(const cv::Mat& depth, int u, int v)
{
    const int r = 1;               // 半径
    std::vector<float> vals;
    vals.reserve(9);

    for (int du = -r; du <= r; ++du)
        for (int dv = -r; dv <= r; ++dv)
        {
            int uu = u + du < 0 ? 0 : (u + du >= depth.cols ? depth.cols - 1 : u + du);
            int vv = v + dv < 0 ? 0 : (v + dv >= depth.rows ? depth.rows - 1 : v + dv);
            float d = depth.at<float>(vv, uu);   // 注意 vv,uu 顺序
            if (std::isfinite(d) && d > 0.f) vals.push_back(d);
        }

    if (vals.empty()) return 0.f;   // 0的情况
    std::nth_element(vals.begin(), vals.begin() + vals.size()/2, vals.end());
    return vals[vals.size()/2];
}

void MultiSensorFusionPerceptionNodelet::legCallback(const people_tracker::LegArray::ConstPtr& leg_msg)
{
    if (!has_uwb_point_ && !has_zed_point_) {
        ROS_WARN_THROTTLE(1.0, "No UWB or ZED point available yet.");
        return;
    }

    double min_dist = std::numeric_limits<double>::max();
    people_tracker::Leg closest_leg;

    for (const auto& leg : leg_msg->legs) {
        if (leg.confidence < 0.9) continue;

        double dx_uwb = leg.position.x - latest_uwb_point_base_.x;
        double dy_uwb = leg.position.y - latest_uwb_point_base_.y;
        double dist_uwb = std::sqrt(dx_uwb * dx_uwb + dy_uwb * dy_uwb);

        double dx_zed = leg.position.x - latest_zed_point_base_.x;
        double dy_zed = leg.position.y - latest_zed_point_base_.y;
        double dist_zed = std::sqrt(dx_zed * dx_zed + dy_zed * dy_zed);

        double dist = std::min(dist_uwb, dist_zed);

        if (dist < min_dist) {
            min_dist = dist;
            closest_leg = leg;
        }
    }


    if (min_dist == std::numeric_limits<double>::max()) return;

    ROS_INFO("Laser in base_link0: x=%.2f, y=%.2f, z=%.2f",closest_leg.position.x, closest_leg.position.y,0.3);

    visualization_msgs::MarkerArray ma;
    visualization_msgs::Marker m;
    m.header = leg_msg->header;
    m.ns = "closest_leg";
    m.id = 0;
    m.type = visualization_msgs::Marker::SPHERE;
    m.action = visualization_msgs::Marker::ADD;
    m.pose.position = closest_leg.position;
    m.pose.position.z = 0.3;
    m.pose.orientation.w = 1.0;
    m.scale.x = m.scale.y = m.scale.z = 0.2;
    m.color.r = 1.0;
    m.color.g = 1.0;
    m.color.b = 0.0;
    m.color.a = 1.0;
    m.lifetime = ros::Duration(1.0);

    ma.markers.push_back(m);
    pub_leg_marker_.publish(ma);
}

// void MultiSensorFusionPerceptionNodelet::legCallback(const people_tracker::LegArray::ConstPtr& leg_msg)
// {
//   visualization_msgs::MarkerArray ma;
//   ma.markers.reserve(leg_msg->legs.size());

//   for (size_t i = 0; i < leg_msg->legs.size(); ++i)
//   {
//      /* ---- 筛选：只保留运动的人腿 ---- */
//     if (leg_msg->legs[i].confidence <0.9) 
//         continue;
//     visualization_msgs::Marker m;
//     m.header       = leg_msg->header;          // 时间戳 / frame_id 与原消息一致
//     m.ns           = "legs";
//     m.id           = i;   // 
//     m.type         = visualization_msgs::Marker::SPHERE;
//     m.action       = visualization_msgs::Marker::ADD;

//     m.pose.position = leg_msg->legs[i].position;
//     m.pose.position.z = 0.3;               // 强制距地面0.3m
//     m.pose.orientation.w = 1.0;

//     m.scale.x = m.scale.y = m.scale.z = 0.2; // 20 cm 小球
//     m.color.r = 0.0; 
//     m.color.g = 1.0; 
//     m.color.b = 0.0; 
//     m.color.a = 0.9;
//     m.lifetime = ros::Duration(1);         // 1s 后自动消失

//     ma.markers.push_back(m);
//   }
//   pub_leg_marker_.publish(ma);
// }

// void MultiSensorFusionPerceptionNodelet::spin(
//     const sensor_msgs::CompressedImageConstPtr& img_msg,
//     const sensor_msgs::ImageConstPtr& depth_msg,
//     const nlink_parser::LinktrackAoaNodeframe0::ConstPtr& uwb_msg,
//     const people_tracker::LegArray::ConstPtr& leg_msg)
// {
//     lookupStaticTransforms();
//     // print_transform_matrix();
//     uwbCallback(uwb_msg);
//     zedCallback(img_msg, depth_msg);
//     legCallback(leg_msg);
//     bool occluded = isOccluded(latest_zed_point_base_, last_zed_detections_);
//     publishOcclusionMarker(occluded, latest_zed_point_base_);
    
//     NODELET_INFO("------------spin finished------------.");
// }
void MultiSensorFusionPerceptionNodelet::spin(
    const sensor_msgs::CompressedImageConstPtr& img_msg,
    const sensor_msgs::ImageConstPtr& depth_msg,
    const nlink_parser::LinktrackAoaNodeframe0::ConstPtr& uwb_msg,
    const people_tracker::LegArray::ConstPtr& leg_msg)
{
    lookupStaticTransforms();
    
    // 处理各个传感器回调
    uwbCallback(uwb_msg);
    zedCallback(img_msg, depth_msg);
    legCallback(leg_msg);
    
    // 更新跟踪器
    geometry_msgs::Point default_point;
    default_point.x = 0;
    default_point.y = 0;
    default_point.z = 0;
    
    person_tracker_->update(
        has_uwb_point_ ? latest_uwb_point_base_ : default_point,
        has_zed_point_ ? latest_zed_point_base_ : default_point,
        default_point, // 这里需要根据legCallback的结果传入正确的腿部位置
        img_msg->header.stamp
    );
    
    // 发布跟踪结果
    publishTrackingResults();
    
    // 遮挡检测
    bool occluded = isOccluded(latest_zed_point_base_, last_zed_detections_);
    publishOcclusionMarker(occluded, latest_zed_point_base_);
    
    NODELET_INFO("------------spin finished------------.");
}

void MultiSensorFusionPerceptionNodelet::lookupStaticTransforms()
{
    try
    {
        // 互为逆变换，一次性拿全
        tf_base2odom_ = tf_buffer_.lookupTransform("odom_est", "base_link0", ros::Time(0));
        tf_odom2base_ = tf_buffer_.lookupTransform("base_link0", "odom_est", ros::Time(0));
        tf_nlink2odom_ = tf_buffer_.lookupTransform("odom_est", "nlink", ros::Time(0));
        tf_nlink2base_ = tf_buffer_.lookupTransform("base_link0", "nlink", ros::Time(0));
        tf_zed2odom_ = tf_buffer_.lookupTransform("odom_est", "zed2_left_camera_optical_frame", ros::Time(0));
        tf_zed2base_ = tf_buffer_.lookupTransform("base_link0","zed2_left_camera_optical_frame", ros::Time(0));
    }
    catch (tf2::TransformException& ex)
    {
        ROS_WARN_THROTTLE(1.0, "[lookupStaticTransforms] %s", ex.what());
    }
}

void MultiSensorFusionPerceptionNodelet::caminfoCallback(const sensor_msgs::CameraInfoConstPtr &caminfo) 
{
    if(caminfo_updated) return;
    else 
    {
        ROS_INFO("Start to get camera_info.");
        transform_i2p = (cv::Mat_<double>(3,4) << caminfo->P.at(0), caminfo->P.at(1), caminfo->P.at(2), caminfo->P.at(3),
                                                  caminfo->P.at(4), caminfo->P.at(5), caminfo->P.at(6), caminfo->P.at(7),
                                                  caminfo->P.at(8), caminfo->P.at(9), caminfo->P.at(10), caminfo->P.at(11));
        transform_c2i = (cv::Mat_<double>(4,4) << 0.000, -1.000, 0.000,  0.000,
                                                  0.000,  0.000,  -1.000,  0.000,
                                                  1.000,  0.000,  0.000,  0.000,
                                                  0.000,  0.000,  0.000,  1.000);  
        transform_u2c = (cv::Mat_<double>(4,4) << 0.998463,    -0.025775,    -0.049055,     0.368415, 
                                                  0.025568,     0.999661,    -0.004836,     0.124722, 
                                                  0.049163,     0.003574,     0.998784,    -0.271747, 
                                                  0.000000,     0.000000,     0.000000,     1.000000 );
        transform_u2p = transform_i2p * transform_c2i * transform_u2c;
        caminfo_updated = true;
        ROS_INFO("Already get camera_info!");
    }
}

void MultiSensorFusionPerceptionNodelet::loadTopicParameters() {
    // 设置默认值
    pnh.param("camera_info_topic", caminfo_topic_, std::string("/zed2/zed_node/rgb/camera_info"));
    pnh.param("image_topic", img_topic_, std::string("/zed2/zed_node/rgb/image_rect_color/compressed"));
    pnh.param("depth_topic", depth_topic_, std::string("/zed2/zed_node/depth/depth_registered"));
    pnh.param("uwb_topic", uwb_topic_, std::string("/uwb_filter_polar"));
    pnh.param("visualization_topic", vis_topic_, std::string("/yolo_imgae_seg"));
    pnh.param("zed_point_topic", zed_point_topic_, std::string("/zed_point"));
    pnh.param("zed_marker_topic", zed_marker_topic_, std::string("/zed_point_marker"));
    pnh.param("uwb_point_topic", uwb_point_topic_, std::string("/uwb_point"));
    pnh.param("uwb_marker_topic", uwb_marker_topic_, std::string("/uwb_point_marker"));
}

// 打印参数配置的函数
void MultiSensorFusionPerceptionNodelet::printTopicConfiguration() {
    ROS_INFO("=== Topic Configuration ===");
    ROS_INFO("  Camera Info: %s", caminfo_topic_.c_str());
    ROS_INFO("  Image: %s", img_topic_.c_str());
    ROS_INFO("  Depth: %s", depth_topic_.c_str());
    ROS_INFO("  UWB: %s", uwb_topic_.c_str());
    ROS_INFO("  Visualization: %s", vis_topic_.c_str());
    ROS_INFO("  ZED Point: %s", zed_point_topic_.c_str());
    ROS_INFO("  ZED Marker: %s", zed_marker_topic_.c_str());
    ROS_INFO("  UWB Point: %s", uwb_point_topic_.c_str());
    ROS_INFO("  UWB Marker: %s", uwb_marker_topic_.c_str());
    ROS_INFO("============================");
}

void MultiSensorFusionPerceptionNodelet::print_transform_matrix()
{
    geometry_msgs::TransformStamped temp;
    std::string frame_id = "zed2_left_camera_frame";
    std::string child_frame_id = "zed2_left_camera_optical_frame";//zed2_left_camera_optical_frame
    try 
    {
        temp = tf_buffer_.lookupTransform(child_frame_id,frame_id,ros::Time(0),ros::Duration(0.1));
    }
    catch (tf2::TransformException& ex) 
    {
        ROS_WARN_THROTTLE(1.0, "%s", ex.what());
        return;
    }

    Eigen::Isometry3d eigen_T = tf2::transformToEigen(temp);
    Eigen::Matrix4d mat = eigen_T.matrix();

    std::ostringstream oss;
    oss << "\n------ "<< child_frame_id <<"<-- "<< frame_id << "------\n";
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j)
            oss << std::setw(12) << std::fixed << std::setprecision(6) << mat(i,j) << " ";
        oss << "\n";
    }
    oss << "----------------------------------------------";
    ROS_INFO_STREAM(oss.str());
}


/*
------ zed2_left_camera_optical_frame<-- nlink------
   -0.025568    -0.999661     0.004836    -0.124722 
   -0.049163    -0.003574    -0.998784     0.271747 
    0.998463    -0.025775    -0.049055     0.368415 
    0.000000     0.000000     0.000000     1.000000 
------ zed2_left_camera_frame<-- nlink------
    0.998463    -0.025775    -0.049055     0.368415 
    0.025568     0.999661    -0.004836     0.124722 
    0.049163     0.003574     0.998784    -0.271747 
    0.000000     0.000000     0.000000     1.000000 
------ zed2_left_camera_optical_frame<-- zed2_left_camera_frame------
    0.000000    -1.000000     0.000000     0.000000 
    0.000000     0.000000    -1.000000     0.000000 
    1.000000     0.000000     0.000000     0.000000 
    0.000000     0.000000     0.000000     1.000000
------ pix <-- nlink------
[[ 264.7289    -6.8364    22.7602  -23.4115 ]
 [ -8.3150    267.1739    14.2528 -142.7931 ]
 [  0.998463  -0.025775  -0.049055   0.368415]]
*/

/* ============ 1. 动态遮挡：2-D  bbox IOU + 深度次序 ============ */
static double bboxArea(const float* b)   // [x1,y1,x2,y2]
{
    return std::max(0.0f, b[2]-b[0]) * std::max(0.0f, b[3]-b[1]);
}
/* 1. 底层实现：只认 float[4] */
static double bboxIOU(const float* a, const float* b)
{
    float xx1 = std::max(a[0], b[0]);
    float yy1 = std::max(a[1], b[1]);
    float xx2 = std::min(a[2], b[2]);
    float yy2 = std::min(a[3], b[3]);
    float inter = std::max(0.0f, xx2-xx1) * std::max(0.0f, yy2-yy1);
    float uni   = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter;
    return uni < 1e-6f ? 0.0 : inter / uni;
}

/* 2. 对外重载：直接传 Detection */
static double bboxIOU(const Detection& da, const Detection& db)
{
    return bboxIOU(da.bbox, db.bbox);   // 现在这里调用的是上面的指针版
}

/* 主函数：返回 true 表示“被遮挡” */
bool MultiSensorFusionPerceptionNodelet::isOccluded(
        const geometry_msgs::Point& target,
        const std::vector<Detection>& all_objs,
        const std::vector<geometry_msgs::Point>& map_polygons)
{
    /* ------ 1. 动态遮挡：找“挡在前面且IOU大”的框 ------ */
    // 先拿到目标在相机下的 2-D 框（你 zedCallback 里算过，这里直接存到成员）
    static cv::Rect2d tgt_rect;      // 由 last_zed_bbox_ 赋值，见下
    static double     tgt_depth;     // 由 last_zed_depth_ 赋值
    for (const auto& d : all_objs)
    {
        if (d.classId != 0) continue;          // 只比行人
        if (bboxIOU(last_zed_detections_[0], d) < 0.3f) continue;
        /* 取中心深度做次序判断 */
        double other_depth = median3x3(depth_, (d.bbox[0]+d.bbox[2])/2,
                                                (d.bbox[1]+d.bbox[3])/2) * 0.001;
        if (other_depth < tgt_depth - 0.3){   // 别人更近 30 cm 就判挡
            std::cout << "检测到动态遮挡" << std::endl;  // ⬅️ 只这里输出
            return true;
        }
    }

    /* ------ 2. 静态遮挡：射线法判断“目标是否被地图多边形挡” ------ */
    if (!map_polygons.empty())
    {
        geometry_msgs::PointStamped cam_pt;
        geometry_msgs::PointStamped base_pt;
        base_pt.header.frame_id = "base_link0";
        base_pt.point = target;
        try{
            tf_buffer_.transform(base_pt, cam_pt, "zed2_left_camera_optical_frame");
        }catch(...){ return false; }

        // 简单柱子/墙面用 3-D 线段-平面相交
        Eigen::Vector3d ray(cam_pt.point.x, cam_pt.point.y, cam_pt.point.z);
        for (size_t i = 0; i < map_polygons.size(); i += 3)   // 每3点一个三角形
        {
            Eigen::Vector3d a(map_polygons[i].x,   map_polygons[i].y,   map_polygons[i].z);
            Eigen::Vector3d b(map_polygons[i+1].x, map_polygons[i+1].y, map_polygons[i+1].z);
            Eigen::Vector3d c(map_polygons[i+2].x, map_polygons[i+2].y, map_polygons[i+2].z);
            // Möller–Trumbore 射线-三角形相交
            const double EPS = 1e-6;
            Eigen::Vector3d edge1 = b - a;
            Eigen::Vector3d edge2 = c - a;
            Eigen::Vector3d h = ray.cross(edge2);
            double det = edge1.dot(h);
            if (std::fabs(det) < EPS) continue;
            double inv_det = 1.0 / det;
            Eigen::Vector3d s = Eigen::Vector3d::Zero() - a;  // 射线起点在原点
            double u = inv_det * s.dot(h);
            if (u < 0.0 || u > 1.0) continue;
            Eigen::Vector3d q = s.cross(edge1);
            double v = inv_det * ray.dot(q);
            if (v < 0.0 || u + v > 1.0) continue;
            double t = inv_det * edge2.dot(q);
            if (t > 0.01 && t < ray.norm()){   // 遮挡体在相机与目标之间
                std::cout << "检测到静态遮挡" << std::endl;  // ⬅️ 只这里输出
                return true;}
        }
    }
    return false;
}

/* ============ 3. 可视化：红=被挡，绿=可见 ============ */
void MultiSensorFusionPerceptionNodelet::publishOcclusionMarker(bool occluded,
                                                               const geometry_msgs::Point& target)
{
    visualization_msgs::Marker m;
    m.header.frame_id = "base_link0";
    m.header.stamp    = ros::Time::now();
    m.ns              = "occlusion_flag";
    m.id              = 0;
    m.type            = visualization_msgs::Marker::SPHERE;
    m.action          = visualization_msgs::Marker::ADD;
    m.pose.position   = target;
    m.pose.orientation.w = 1.0;
    m.scale.x = m.scale.y = m.scale.z = 0.25;
    m.color.r = occluded ? 1.0 : 0.0;
    m.color.g = occluded ? 0.0 : 1.0;
    m.color.b = 0.0;
    m.color.a = 0.8;
    m.lifetime = ros::Duration(0.2);
    pub_uwb_marker_.publish(m);   // 复用你原来的 marker pub
}

// 跟踪器
void MultiSensorFusionPerceptionNodelet::initializeTracker()
{
    person_tracker_ = std::make_unique<PersonTracker>(nh);
    if (person_tracker_->initialize()) {
        NODELET_INFO("Person tracker initialized successfully");
    } else {
        NODELET_ERROR("Failed to initialize person tracker");
    }
}

void MultiSensorFusionPerceptionNodelet::publishTrackingResults()
{
    // 发布跟踪位置
    auto tracked_pose = person_tracker_->getTrackedPose();
    pub_tracked_person_.publish(tracked_pose);
    
    // 发布跟踪状态
    std_msgs::String status_msg;
    status_msg.data = person_tracker_->getTrackingStatus();
    pub_tracking_status_.publish(status_msg);
    
    // 发布速度信息
    auto velocity = person_tracker_->getVelocity();
    geometry_msgs::Vector3Stamped velocity_msg;
    velocity_msg.header.stamp = ros::Time::now();
    velocity_msg.header.frame_id = "base_link0";
    velocity_msg.vector = velocity;
    pub_person_velocity_.publish(velocity_msg);  // 添加这行来发布速度
}


PLUGINLIB_EXPORT_CLASS(multi_sensor_fusion::MultiSensorFusionPerceptionNodelet, nodelet::Nodelet)

} // namespace multi_sensor_fusion

