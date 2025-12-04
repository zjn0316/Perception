// v2,2025/12/4
#include "multi_sensor_fusion_perception_nodelet_v2.h"
#include <pluginlib/class_list_macros.h>

namespace multi_sensor_fusion {

MultiSensorFusionPerceptionNodelet::MultiSensorFusionPerceptionNodelet():caminfo_updated(false){}

MultiSensorFusionPerceptionNodelet::~MultiSensorFusionPerceptionNodelet() {}

void MultiSensorFusionPerceptionNodelet::onInit() {
    setlocale(LC_ALL,"");
    NODELET_INFO("正在初始化 Multi-Sensor Fusion Perception Nodelet...");
    // 计时开始
    auto start = ros::Time::now();

    // 初始化节点句柄
    nh = getNodeHandle();
    pnh = getPrivateNodeHandle();

    // 加载话题参数
    loadTopicParameters();
    printTopicConfiguration();
    
    // tf监听器
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
    sub_leg_.reset(new message_filters::Subscriber<people_tracker::PersonArray>(nh, "/people_tracked", 2));
    pub_leg_marker_ = nh.advertise<visualization_msgs::MarkerArray>("/leg_point_marker", 1);

    // spin
    sync_.reset(new message_filters::Synchronizer<ApproxSyncPolicy>(ApproxSyncPolicy(50), *sub_img_, *sub_depth_, *sub_uwb_, *sub_leg_));
    sync_->registerCallback(boost::bind(&MultiSensorFusionPerceptionNodelet::spin, this, _1, _2, _3,_4));

    // 计时结束
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


void MultiSensorFusionPerceptionNodelet::legCallback(const people_tracker::PersonArray::ConstPtr& leg_msg)
{
  visualization_msgs::MarkerArray ma;
  ma.markers.reserve(leg_msg->people.size());

  for (size_t i = 0; i < leg_msg->people.size(); ++i)
  {
     /* ---- 筛选：只保留运动的人腿 ---- */
    visualization_msgs::Marker m;
    m.header       = leg_msg->header;          // 时间戳 / frame_id 与原消息一致
    m.ns           = "people";
    m.id           = i;   // 
    m.type         = visualization_msgs::Marker::SPHERE;
    m.action       = visualization_msgs::Marker::ADD;

    m.pose.position = leg_msg->people[i].pose.position;
    m.pose.position.z = 0.3;               // 强制距地面0.3m
    m.pose.orientation.w = 1.0;

    m.scale.x = m.scale.y = m.scale.z = 0.2; // 20 cm 小球
    m.color.r = 0.0; 
    m.color.g = 1.0; 
    m.color.b = 0.0; 
    m.color.a = 0.9;
    m.lifetime = ros::Duration(1);         // 1s 后自动消失

    ma.markers.push_back(m);
  }
  pub_leg_marker_.publish(ma);
}


void MultiSensorFusionPerceptionNodelet::spin(
    const sensor_msgs::CompressedImageConstPtr& img_msg,
    const sensor_msgs::ImageConstPtr& depth_msg,
    const nlink_parser::LinktrackAoaNodeframe0::ConstPtr& uwb_msg,
    const people_tracker::PersonArray::ConstPtr& leg_msg)
{
    ROS_INFO("IMG: %.6f  DEPTH: %.6f  UWB: %.6f  LEG: %.6f",
         img_msg->header.stamp.toSec(),
         depth_msg->header.stamp.toSec(),
         uwb_msg->header.stamp.toSec(),
         leg_msg->header.stamp.toSec());
    lookupStaticTransforms();
    
    // 处理各个传感器回调
    uwbCallback(uwb_msg);
    zedCallback(img_msg, depth_msg);
    legCallback(leg_msg);
    
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
    // 图像-像素、相机-图像、UWB-相机 的变换矩阵
    if(caminfo_updated) return;
    else 
    {
        ROS_INFO("Start to get camera_info.");
        transform_i2p = (cv::Mat_<double>(3,4) << caminfo->P.at(0), caminfo->P.at(1), caminfo->P.at(2), caminfo->P.at(3),
                                                  caminfo->P.at(4), caminfo->P.at(5), caminfo->P.at(6), caminfo->P.at(7),
                                                  caminfo->P.at(8), caminfo->P.at(9), caminfo->P.at(10), caminfo->P.at(11));
        // 从TF里读取出来的，直接硬编码
        transform_c2i = (cv::Mat_<double>(4,4) << 0.000, -1.000, 0.000,  0.000,
                                                  0.000,  0.000,  -1.000,  0.000,
                                                  1.000,  0.000,  0.000,  0.000,
                                                  0.000,  0.000,  0.000,  1.000);  
        transform_u2c = (cv::Mat_<double>(4,4) << 0.998463,    -0.025775,    -0.049055,     0.368415, 
                                                  0.025568,     0.999661,    -0.004836,     0.124722, 
                                                  0.049163,     0.003574,     0.998784,    -0.271747, 
                                                  0.000000,     0.000000,     0.000000,     1.000000 );
        transform_u2p = transform_i2p * transform_c2i * transform_u2c;
        // TODO: 激光雷达的变换矩阵
        caminfo_updated = true;
        ROS_INFO("Already get camera_info!");
    }
}

void MultiSensorFusionPerceptionNodelet::loadTopicParameters() {
    // 从参数服务器读取 8 个话题名，若用户没给就填默认值
    pnh.param("camera_info_topic", caminfo_topic_, std::string("/zed2/zed_node/rgb/camera_info"));
    pnh.param("image_topic", img_topic_, std::string("/zed2/zed_node/rgb/image_rect_color/compressed"));
    pnh.param("depth_topic", depth_topic_, std::string("/zed2/zed_node/depth/depth_registered"));
    pnh.param("uwb_topic", uwb_topic_, std::string("/uwb_filter_polar"));
    pnh.param("visualization_topic", vis_topic_, std::string("/yolo_imgae_seg"));
    pnh.param("zed_point_topic", zed_point_topic_, std::string("/zed_point"));
    pnh.param("zed_marker_topic", zed_marker_topic_, std::string("/zed_point_marker"));
    pnh.param("uwb_point_topic", uwb_point_topic_, std::string("/uwb_point"));
    pnh.param("uwb_marker_topic", uwb_marker_topic_, std::string("/uwb_point_marker"));
    // TODO: 可以添加更多参数
    // target_person(target_person_id,target_person_position,target_person_conf)
    // other_person(other_person_id,other_person_position,other_person_conf)
}

// 打印参数配置的函数
void MultiSensorFusionPerceptionNodelet::printTopicConfiguration() {
    ROS_INFO("===== Topic Configuration ======");
    ROS_INFO("  Camera Info: %s", caminfo_topic_.c_str());
    ROS_INFO("  Image: %s", img_topic_.c_str());
    ROS_INFO("  Depth: %s", depth_topic_.c_str());
    ROS_INFO("  UWB: %s", uwb_topic_.c_str());
    ROS_INFO("  Visualization: %s", vis_topic_.c_str());
    ROS_INFO("  ZED Point: %s", zed_point_topic_.c_str());
    ROS_INFO("  ZED Marker: %s", zed_marker_topic_.c_str());
    ROS_INFO("  UWB Point: %s", uwb_point_topic_.c_str());
    ROS_INFO("  UWB Marker: %s", uwb_marker_topic_.c_str());
    ROS_INFO("================================");
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



PLUGINLIB_EXPORT_CLASS(multi_sensor_fusion::MultiSensorFusionPerceptionNodelet, nodelet::Nodelet)

} // namespace multi_sensor_fusion

