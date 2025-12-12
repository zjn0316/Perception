// v2,2025/12/12
#include "multi_sensor_fusion_perception_nodelet_v9.h"
#include <pluginlib/class_list_macros.h>
#include "cnpy.h"
#include <sys/stat.h>        // mkdir
#include <inttypes.h>        // PRIu64
#include <errno.h>


// 全局计数器，让文件名不冲突
static uint64_t g_frame_id   = 0;   // 全局帧计数（永不断号）
static uint64_t g_seq_id     = 0;   // 当前 seq 序号
static size_t   g_frames_in_seq = 0; // 当前 seq 已收帧数
namespace multi_sensor_fusion {
MultiSensorFusionPerceptionNodelet::MultiSensorFusionPerceptionNodelet():caminfo_updated(false){}
MultiSensorFusionPerceptionNodelet::~MultiSensorFusionPerceptionNodelet() {}

void MultiSensorFusionPerceptionNodelet::onInit() {
    setlocale(LC_ALL,"");
    if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug)) {
        ros::console::notifyLoggerLevelsChanged();
    }
    ROS_INFO("正在初始化 Multi-Sensor Fusion Perception Nodelet...");
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
    caminfo_sub_ = nh.subscribe(caminfo_topic_, 10, &MultiSensorFusionPerceptionNodelet::caminfoCallback, this);

    // uwb
    sub_uwb_.reset(new message_filters::Subscriber<nlink_parser::LinktrackAoaNodeframe0>(nh, uwb_topic_, 10));
    pub_uwb_ = nh.advertise<geometry_msgs::PointStamped>(uwb_point_topic_, 1);
    pub_uwb_marker_ = nh.advertise<visualization_msgs::Marker>(uwb_marker_topic_, 1);

    // zed
    yolo_.reset(new YoloDetector(yolo_engine_path_));
    // 使用 TensorRT 默认 Logger  ， deepsort强制要求输入logger级别
    static TensorRTLogger logger;  // 使用静态确保生命周期
    deepsort_.reset(new DeepSort(deepsort_engine_path_, 32, 256, 0, &logger,
                                 deepsort_max_age_, deepsort_n_init_, deepsort_max_iou_, deepsort_max_cos_)) ;

    sub_img_.reset(new message_filters::Subscriber<sensor_msgs::CompressedImage>(nh,img_topic_, 1));
    sub_depth_.reset(new message_filters::Subscriber<sensor_msgs::Image>(nh,depth_topic_, 1)); 
    pub_yolo_ = nh.advertise<sensor_msgs::Image>(vis_topic_, 1);
    pub_zed_ = nh.advertise<geometry_msgs::PointStamped>(zed_point_topic_, 1);
    pub_zed_marker_ = nh.advertise<visualization_msgs::MarkerArray>(zed_marker_topic_, 1);

    // leg
    sub_leg_.reset(new message_filters::Subscriber<people_tracker::PersonArray>(nh, leg_topic_, 2));
    pub_leg_ = nh.advertise<geometry_msgs::PointStamped>(leg_point_topic_, 1);
    pub_leg_marker_ = nh.advertise<visualization_msgs::MarkerArray>(leg_marker_topic_, 1);

    // fusion
    fusion_.reset(new fusion_net::FusionNetTRT(fusion_engine_path_));
    seq_buf_.resize(12);
    pub_fusion_        = nh.advertise<geometry_msgs::PointStamped>(fusion_point_topic_, 1);
    pub_fusion_marker_ = nh.advertise<visualization_msgs::Marker>(fusion_marker_topic_, 1);

    // spin
    sync_.reset(new message_filters::Synchronizer<ApproxSyncPolicy>(ApproxSyncPolicy(50), *sub_img_, *sub_depth_, *sub_uwb_, *sub_leg_));
    sync_->registerCallback(boost::bind(&MultiSensorFusionPerceptionNodelet::spin, this, _1, _2, _3,_4));

    // 计时结束
    auto duration = (ros::Time::now() - start).toSec();
    ROS_INFO("onInit 花费 %.1f ms. Nodelet已成功初始化！", duration * 1000);
}




void MultiSensorFusionPerceptionNodelet::spin(
    const sensor_msgs::CompressedImageConstPtr& img_msg,
    const sensor_msgs::ImageConstPtr& depth_msg,
    const nlink_parser::LinktrackAoaNodeframe0::ConstPtr& uwb_msg,
    const people_tracker::PersonArray::ConstPtr& leg_msg)
{
    ros::Time t0, t8;   
    t0 = ros::Time::now();
    lookupStaticTransforms();
    // debug 打印变换矩阵示例：
    // print_transform_matrix("nlink", "zed2_left_camera_optical_frame");
    /* ====== 每帧重置门控 ====== */
    best_leg_d2_  = 0.1;
    best_zed_d2_  = 0.2;
    best_leg_idx_ = -1;
    best_zed_idx_ = -1;
    target_uwb_ = geometry_msgs::PointStamped{}; 
    target_leg_ = geometry_msgs::PointStamped{};   // 全 0
    target_zed_ = geometry_msgs::PointStamped{};


    // 处理各个传感器回调
    /* ======================== 1. UWB 处理 ========================== */
    /* -------------- 1. 有效性检查 -------------- */
    if (uwb_msg->nodes.empty() || uwb_msg->nodes.at(0).role != 1)
        return;

    // -------------- 2. 计算UWB置信度 -------------- */
    float fp_rssi = uwb_msg->nodes.at(0).fp_rssi;
    float rx_rssi = uwb_msg->nodes.at(0).rx_rssi;
    float dis = uwb_msg->nodes.at(0).dis;// 距离
    float ang = uwb_msg->nodes.at(0).angle * M_PI / 180.0f;// 角度
    float uwb_conf_ = calcUwbConfidence(fp_rssi, rx_rssi, dis);

    /* -------------- 3. 计算各坐标系下坐标 -------------- */
    geometry_msgs::PointStamped uwb_pt_nlink;// uwb在nlink坐标系下的坐标
    uwb_pt_nlink.header.stamp    = uwb_msg->header.stamp;
    uwb_pt_nlink.header.frame_id = "nlink";
    uwb_pt_nlink.point.x = dis * std::cos(ang);
    uwb_pt_nlink.point.y = dis * std::sin(ang);
    uwb_pt_nlink.point.z = 0.45;// nlink坐标系下uwb标签大约高度 45 m

    geometry_msgs::PointStamped uwb_pt_base;// uwb在base_link0坐标系下的坐标
    tf2::doTransform(uwb_pt_nlink, uwb_pt_base, tf_nlink2base_);

    geometry_msgs::PointStamped uwb_pt_odom;// uwb在odom_est坐标系下的坐标
    tf2::doTransform(uwb_pt_nlink, uwb_pt_odom, tf_base2odom_);  
    
    geometry_msgs::PointStamped uwb_pt_cam;// uwb在相机坐标系下的坐标
    tf2::doTransform(uwb_pt_nlink, uwb_pt_cam, tf_nlink2cam_);

    uwb_uv_ = camera2pixel(uwb_pt_cam, K);// uwb在像素坐标系下的坐标
    // debug 打印 UWB 所有坐标系下坐标
    printUwbPositions(uwb_pt_base, uwb_pt_odom, uwb_pt_cam, uwb_uv_);
    // ROS_INFO("UWB uwb_conf=%.2f ", uwb_conf_);

    /* -------------- 4. 发布话题 -------------- */
    visualization_msgs::Marker marker;
    marker.header           = uwb_pt_base.header;
    marker.ns               = "uwb";
    marker.id               = 0;
    marker.type             = visualization_msgs::Marker::SPHERE;
    marker.action           = visualization_msgs::Marker::ADD;
    marker.pose.position    = uwb_pt_base.point;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = marker.scale.y = marker.scale.z = 0.2;
    marker.color.r = 0.0f;
    marker.color.g = 0.0f;
    marker.color.b = 1.0f;// 蓝色
    marker.color.a = uwb_conf_;
    marker.lifetime = ros::Duration(1.0);// 持续 1 秒
    pub_uwb_marker_.publish(marker);
    target_uwb_ = uwb_pt_base;          // UWB 始终有





    /* ======================== 2. Leg 处理 ========================== */
    visualization_msgs::MarkerArray ma_leg;
    ma_leg.markers.reserve(leg_msg->people.size());

    std::vector<cv::Point2d> leg_uvs; 
    leg_uvs.reserve(leg_msg->people.size());
    for (size_t i = 0; i < leg_msg->people.size(); ++i)
    {
        const auto& p = leg_msg->people[i];
        if (p.pose.position.z < 0.5) continue;   // 只画运动的人（z=1.0）

        /* -------------- 1. 计算leg在各坐标系下坐标 -------------- */
        geometry_msgs::PointStamped leg_pt_odom;// leg 在 odom_est 下的坐标
        leg_pt_odom.header = leg_msg->header;
        leg_pt_odom.point = p.pose.position;

        geometry_msgs::PointStamped leg_pt_base;// leg 在 base_link0 下的坐标
        tf2::doTransform(leg_pt_odom, leg_pt_base, tf_odom2base_);

        geometry_msgs::PointStamped leg_pt_cam;// leg 在 cam 下的坐标
        tf2::doTransform(leg_pt_base, leg_pt_cam, tf_base2cam_);

        cv::Point2d leg_uv = camera2pixel(leg_pt_cam, K);// leg 在 pixel 下的坐标
        leg_uvs.push_back(leg_uv); 
        printLegPositions(leg_pt_odom, leg_pt_base, leg_pt_cam, leg_uv);
        

        float dx = leg_pt_base.point.x - uwb_pt_base.point.x;
        float dy = leg_pt_base.point.y - uwb_pt_base.point.y;
        float d2 = dx*dx + dy*dy ;
        if (d2 < best_leg_d2_) {          // best_leg_d2_ 初始值 = max_assoc_dist2_
            best_leg_d2_  = d2;
            best_leg_idx_ = static_cast<int>(i);   // 记录最佳下标
            target_leg_   = leg_pt_base ;
            leg_conf_     = p.confidence;
            leg_uv_       = leg_uv;
        }
        
        /* -------------- 2. 发布marker -------------- */
        visualization_msgs::Marker m;
        m.header       = leg_pt_base.header;
        m.ns           = "moving_people";
        m.id           = p.id;
        m.type         = visualization_msgs::Marker::SPHERE;
        m.action       = visualization_msgs::Marker::MODIFY;
        m.pose.position = leg_pt_base.point;
        m.pose.position.z = 0.3;  // laser 扫描面大约 0.3 m 高度
        m.pose.orientation.w = 1.0;
        m.scale.x = m.scale.y = m.scale.z = 0.2;
        m.color.r = 0.0;
        m.color.g = 1.0;
        m.color.b = 0.0;
        m.color.a = p.confidence;  // 透明度表示置信度
        m.lifetime = ros::Duration(1.0);   // 1s 后自动消失，避免拖影
        ma_leg.markers.push_back(m);
    }
    pub_leg_marker_.publish(ma_leg);








    /* ======================== 3. ZED 处理 ========================== */
    /* -------------- 1. 预处理：解压图片 -------------- */
    cv_bridge::CvImagePtr cv_ptr_zed;
    try {
        cv_ptr_zed = cv_bridge::toCvCopy(img_msg, "bgr8");
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    cv::Mat img_zed = cv_ptr_zed->image;

    cv::Mat depth_zed;
    try {
        depth_zed = cv_bridge::toCvShare(depth_msg,sensor_msgs::image_encodings::TYPE_32FC1)->image;
    } catch (cv_bridge::Exception& e) {
        NODELET_ERROR("depth cv_bridge exception: %s", e.what());
        return;
    }


    /* -------------- 2. Yolo + Deepsort -------------- */
    std::vector<Detection> res    = yolo_->inference(img_zed);
    std::vector<Detection> res_fd = filterResult(res);

    std::vector<DetectBox> dets_deepsort = toDetectBoxPed(res_fd); // 转换为 DeepSort 格式
    deepsort_->sort(img_zed, dets_deepsort);   // 内部已做特征提取+数据关联

    
    /* -------------- 3.计算各坐标系下坐标 -------------- */
    std::vector<cv::Point2d> zed_uvs;          // 每帧临时存所有 zed 像素坐标
    zed_uvs.reserve(dets_deepsort.size());      // 申请内存

    visualization_msgs::MarkerArray marker_array_zed;
    for (size_t i = 0; i < dets_deepsort.size(); ++i) {
        const auto& d = dets_deepsort[i];          // DeepSort 结果
        const auto& yoloDet = res_fd[i];           // 同一索引的 YOLO 结果（含掩码）
        int trackID = d.trackID;

        // 计算置信度
        float mid_depth = depthInsideMask(yoloDet.maskMatrix,depth_zed,cv::Rect(d.x1, d.y1, d.x2-d.x1, d.y2-d.y1));
        MaskDepthStats stats = maskDepthStats(yoloDet.maskMatrix, depth_zed,cv::Rect(d.x1, d.y1, d.x2-d.x1, d.y2-d.y1));
        float zed_conf = calcZedConfidence(d.confidence,stats.validRatio,stats.varDepth);
        // ROS_INFO("ZED zed_conf=%.2f ", zed_conf);

        // 计算目标中心像素坐标
        int zed_u = std::clamp(static_cast<int>((d.x1 + d.x2) * 0.5f), 0, depth_zed.cols - 1);
        int zed_v = std::clamp(static_cast<int>((d.y1 + d.y2) * 0.5f), 0, depth_zed.rows - 1);
        zed_uv_=cv::Point2d(zed_u,zed_v);
        // 根据索引得到深度值
        float depth_raw = median3x3(depth_zed, zed_u, zed_v);
        if (depth_raw >= 15000 || depth_raw <= 0.f) continue;
        float depth = depth_raw * 0.001f;
        if (std::abs(depth - mid_depth) > 0.5f) continue;
        // ROS_INFO("TrackID=%d mid_depth=%.1f m  depth=%.1f m", trackID, mid_depth, depth);

        cv::Point2d zed_uv(zed_u, zed_v);// zed在像素坐标系下的坐标
        zed_uvs.push_back(zed_uv); 

        cv::Vec3f zed_point_in_camera = pixel2camera(zed_uv, depth, K);
        geometry_msgs::PointStamped zed_pt_cam;// zed在相机坐标系下的坐标
        zed_pt_cam.header.stamp    = depth_msg->header.stamp;
        zed_pt_cam.header.frame_id = "zed2_left_camera_optical_frame";
        zed_pt_cam.point.x         = zed_point_in_camera[0];
        zed_pt_cam.point.y         = zed_point_in_camera[1];
        zed_pt_cam.point.z         = zed_point_in_camera[2];

        geometry_msgs::PointStamped zed_pt_base;// zed在base_link0坐标系下的坐标
        tf2::doTransform(zed_pt_cam, zed_pt_base, tf_zed2base_);  
        
        geometry_msgs::PointStamped zed_pt_odom;// zed在odom_est坐标系下的坐标
        tf2::doTransform(zed_pt_cam, zed_pt_odom, tf_zed2odom_);  
        printZedPositions(zed_pt_cam, zed_pt_base, zed_pt_odom, zed_uv, d.trackID);

        float dx = zed_pt_base.point.x - uwb_pt_base.point.x;
        float dy = zed_pt_base.point.y - uwb_pt_base.point.y;
        float d2 = dx*dx + dy*dy ;
        if (d2 < best_zed_d2_) {
            best_zed_d2_  = d2;
            best_zed_idx_ = static_cast<int>(i);
            target_zed_  = zed_pt_base; 
            zed_conf_    = zed_conf;
            zed_uv_      = zed_uv;
        }

        visualization_msgs::Marker marker;
        marker.header                = zed_pt_base.header;
        marker.ns                    = "zed";
        marker.id                    = d.trackID;
        marker.type                  = visualization_msgs::Marker::SPHERE;
        marker.action                = visualization_msgs::Marker::ADD;
        marker.pose.position         = zed_pt_base.point;
        marker.pose.orientation.w    = 1.0;
        marker.scale.x = marker.scale.y = marker.scale.z = 0.2;
        marker.color.r = 1.0f;// 红色
        marker.color.g = 0.0f;
        marker.color.b = 0.0f;
        marker.color.a = zed_conf;
        marker.lifetime = ros::Duration(1.0);
        marker_array_zed.markers.push_back(marker);
    }
    pub_zed_marker_.publish(marker_array_zed);




    /* -------------- 6. 可视化发布图片 -------------- */
    cv::Mat& img_seg = cv_ptr_zed->image;
    // 画框图
    YoloDetector::draw_image(img_seg, res_fd);
    // 画
    for (size_t i = 0; i < res_fd.size(); ++i) {
        const auto& d = res_fd[i];
        if (d.classId != 0) continue;              // 只画 person
        /*  ID */
        cv::Scalar color = cv::Scalar(0, 255, 0);                  // 可改成根据 ID 选颜色
        cv::rectangle(img_seg,cv::Point(d.bbox[0], d.bbox[1]),cv::Point(d.bbox[2], d.bbox[3]),color, 2);
        cv::putText(img_seg,"ID:" + std::to_string((int)dets_deepsort[i].trackID),
                    cv::Point(d.bbox[0], d.bbox[1] - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
    }
    // 画 UWB 蓝点
    cv::circle(img_seg, uwb_uv_, 5, cv::Scalar(255, 0, 0), -1);
    // 画 Leg 绿点
    cv::circle(img_seg, leg_uv_, 5, cv::Scalar(0, 255, 0), -1);
    // 画 ZED 红点 
    cv::circle(img_seg, zed_uv_, 5, cv::Scalar(0, 0, 255), -1);
    
    sensor_msgs::ImagePtr msg_seg =cv_bridge::CvImage(img_msg->header, "bgr8", img_seg).toImageMsg();
    pub_yolo_.publish(msg_seg);
    t8 = ros::Time::now();


    // // 安全取 ZED id
    // int zed_id = -1;
    // if (!dets_deepsort.empty() && best_zed_idx_ >= 0 &&best_zed_idx_ < static_cast<int>(dets_deepsort.size()))
    //     zed_id = dets_deepsort[best_zed_idx_].trackID;
    // /* ===================== 融合阶段 ===================== */
    // ROS_INFO("\nUWB=(%.2f,%.2f,%.2f)\n"
    //          "Leg=(%.2f,%.2f,%.2f)\t   id=%d\n"
    //          "ZED=(%.2f,%.2f,%.2f)\t   id=%d",
    //          target_uwb_.point.x, target_uwb_.point.y, target_uwb_.point.z,
    //          target_leg_.point.x, target_leg_.point.y, target_leg_.point.z,
    //          best_leg_idx_ >= 0 ? leg_msg->people[best_leg_idx_].id : -1,   // ✅ 当场拿 id
    //          target_zed_.point.x, target_zed_.point.y, target_zed_.point.z,zed_id);
    print_positions(target_uwb_,target_leg_,target_zed_);
    print_confidences(uwb_conf_,leg_conf_,zed_conf_);


    // fusion
    pushSensorToSeq(target_uwb_, target_leg_, target_zed_,
                    uwb_conf_,   leg_conf_,   zed_conf_);
    doFusionInferAndPub();
    ROS_INFO("spin 花费 %.1f ms.------------spin finished------------.", (t8-t0).toSec() * 1000);
}



// =========================================== 话题 ================================================
void MultiSensorFusionPerceptionNodelet::loadTopicParameters() {
    // 从参数服务器读取话题名，若用户没给就填默认值
    // 订阅
    pnh.param("uwb_topic", uwb_topic_, std::string("/uwb_filter_polar"));
    pnh.param("camera_info_topic", caminfo_topic_, std::string("/zed2/zed_node/rgb/camera_info"));
    pnh.param("image_topic", img_topic_, std::string("/zed2/zed_node/rgb/image_rect_color/compressed"));
    pnh.param("depth_topic", depth_topic_, std::string("/zed2/zed_node/depth/depth_registered"));
    pnh.param("leg_topic",         leg_topic_,         std::string("/people_tracked"));
    // 发布
    pnh.param("uwb_point_topic", uwb_point_topic_, std::string("/uwb_point"));
    pnh.param("uwb_marker_topic", uwb_marker_topic_, std::string("/uwb_point_marker"));
    pnh.param("zed_point_topic", zed_point_topic_, std::string("/zed_point"));
    pnh.param("zed_marker_topic", zed_marker_topic_, std::string("/zed_point_marker"));
    pnh.param("visualization_topic", vis_topic_, std::string("/yolo_imgae_seg"));
    pnh.param("leg_point_topic",  leg_point_topic_,  std::string("/leg_point"));
    pnh.param("leg_marker_topic",  leg_marker_topic_,  std::string("/leg_point_marker"));
    pnh.param("fusion_point_topic",  fusion_point_topic_,  std::string("/fusion_point"));
    pnh.param("fusion_marker_topic",  fusion_marker_topic_,  std::string("/fusion_point_marker"));

    // 路径参数
    pnh.param("yolo_engine_path", yolo_engine_path_,std::string("/home/zjn/catkin_ws/src/multi_sensor_fusion_perception_nodelet/engine/yolo11s-seg.plan"));
    pnh.param("deepsort_engine_path", deepsort_engine_path_,std::string("/home/zjn/catkin_ws/src/multi_sensor_fusion_perception_nodelet/engine/deepsort.engine"));
    pnh.param("fusion_engine_path", fusion_engine_path_,std::string("/home/zjn/catkin_ws/src/multi_sensor_fusion_perception_nodelet/engine/fusion_net_3d.engine"));
    // 其他参数
    pnh.param("yolo_conf_thresh", yolo_conf_threshold_, 0.4);          // 默认 0.3
    pnh.param("deepsort_max_age",  deepsort_max_age_, 200);          // 默认 200 帧
    pnh.param("deepsort_n_init",   deepsort_n_init_, 3);
    pnh.param("deepsort_max_iou",  deepsort_max_iou_, 0.7);
    pnh.param("deepsort_max_cos",  deepsort_max_cos_, 0.2);
}

// 打印参数配置的函数
void MultiSensorFusionPerceptionNodelet::printTopicConfiguration() {
    ROS_INFO("============== 订阅 =============");
    ROS_INFO("  UWB: %s", uwb_topic_.c_str());
    ROS_INFO("  Camera Info: %s", caminfo_topic_.c_str());
    ROS_INFO("  Image: %s", img_topic_.c_str());
    ROS_INFO("  Depth: %s", depth_topic_.c_str());
    ROS_INFO("  Leg: %s", leg_topic_.c_str());
    ROS_INFO("============== 发布 =============");
    ROS_INFO("  UWB Point: %s", uwb_point_topic_.c_str());
    ROS_INFO("  UWB Marker: %s", uwb_marker_topic_.c_str());
    ROS_INFO("  ZED Point: %s", zed_point_topic_.c_str());
    ROS_INFO("  ZED Marker: %s", zed_marker_topic_.c_str());
    ROS_INFO("  Visualization: %s", vis_topic_.c_str());
    ROS_INFO("  Leg Point: %s", leg_point_topic_.c_str());
    ROS_INFO("  Leg Marker: %s", leg_marker_topic_.c_str());
    ROS_INFO("  Fusion Point: %s", fusion_point_topic_.c_str());
    ROS_INFO("  Fuison Marker: %s", fusion_marker_topic_.c_str());
    ROS_INFO("============== 路径 =============");
    ROS_INFO("  YOLO Engine Path: %s", yolo_engine_path_.c_str());
    ROS_INFO("  DeepSort Engine Path: %s", deepsort_engine_path_.c_str());
    ROS_INFO("  Fuison Engine Path: %s", fusion_engine_path_.c_str());
    ROS_INFO("============ 其他参数 ============");
    ROS_INFO("  Yolo Conf Thresh: %.2f", yolo_conf_threshold_);
    ROS_INFO("  DeepSort Max Age: %d",   deepsort_max_age_);
    ROS_INFO("  DeepSort N Init:  %d",   deepsort_n_init_);
    ROS_INFO("  DeepSort Max IoU: %.2f", deepsort_max_iou_);
    ROS_INFO("  DeepSort Max Cos: %.2f", deepsort_max_cos_);
    ROS_INFO("================================");
}

// =========================================== TF ================================================
void MultiSensorFusionPerceptionNodelet::lookupStaticTransforms()
{
    try
    {
        // 互为逆变换，一次性拿全
        tf_nlink2cam_ = tf_buffer_.lookupTransform("zed2_left_camera_optical_frame", "nlink", ros::Time(0));
        tf_nlink2odom_ = tf_buffer_.lookupTransform("odom_est", "nlink", ros::Time(0));
        tf_nlink2base_ = tf_buffer_.lookupTransform("base_link0", "nlink", ros::Time(0));

        tf_zed2odom_ = tf_buffer_.lookupTransform("odom_est", "zed2_left_camera_optical_frame", ros::Time(0));
        tf_zed2base_ = tf_buffer_.lookupTransform("base_link0","zed2_left_camera_optical_frame", ros::Time(0));

        tf_base2odom_ = tf_buffer_.lookupTransform("odom_est", "base_link0", ros::Time(0));
        tf_odom2base_ = tf_buffer_.lookupTransform("base_link0", "odom_est", ros::Time(0));

        tf_base2cam_ = tf_buffer_.lookupTransform("zed2_left_camera_optical_frame","base_link0", ros::Time(0));
    }
    catch (tf2::TransformException& ex)
    {
        ROS_WARN_THROTTLE(1.0, "[lookupStaticTransforms] %s", ex.what());
    }
}

// =========================================== caminfo ================================================
void MultiSensorFusionPerceptionNodelet::caminfoCallback(const sensor_msgs::CameraInfoConstPtr &caminfo) 
{
    // 图像-像素、相机-图像、UWB-相机 的变换矩阵
    if(caminfo_updated) return;
    else 
    {
        ROS_INFO("Start to get camera_info.");
        P = (cv::Mat_<double>(3,4) << caminfo->P.at(0), caminfo->P.at(1), caminfo->P.at(2), caminfo->P.at(3),
                                      caminfo->P.at(4), caminfo->P.at(5), caminfo->P.at(6), caminfo->P.at(7),
                                      caminfo->P.at(8), caminfo->P.at(9), caminfo->P.at(10), caminfo->P.at(11));
        K = (cv::Mat_<double>(3, 3) << caminfo->P[0], caminfo->P[1], caminfo->P[2],
                                       caminfo->P[4], caminfo->P[5], caminfo->P[6],
                                       caminfo->P[8], caminfo->P[9], caminfo->P[10]);
        // ROS_INFO_STREAM("投影矩阵 P: \n" << P);
        // ROS_INFO_STREAM("内参矩阵 K: \n" << K);
        caminfo_updated = true;
        ROS_INFO("Already get camera_info!");
    }
}

// =========================================== UWB ================================================
float MultiSensorFusionPerceptionNodelet::calcUwbConfidence(float fp_rssi,
                                                            float rx_rssi,
                                                            float dis) const
{
    float signal_score = std::clamp((fp_rssi + 180.f) / 100.f, 0.f, 1.f);// 信号强度得分：-80 dBm→1，-180 dBm→0
    float multipath_score = std::clamp(1 - (rx_rssi - fp_rssi) / 30.f, 0.f, 1.f);// 多径效应得分：若 fp_rssi ≪ rx_rssi → 多径严重，置信度应下降
    float dis_score = std::clamp(1 - (dis - 3.f) / 3.f, 0.f, 1.f);// 距离得分：0-3 m→1，3-6 m→1→0，>6 m→0
    return std::clamp(signal_score * multipath_score * dis_score, 0.f, 1.f);
}

// 坐标变换函数
cv::Point2d MultiSensorFusionPerceptionNodelet::camera2pixel(const geometry_msgs::PointStamped& pt_cam,const cv::Matx33d& K){
    // 归一化相机坐标
    double x_n = pt_cam.point.x / pt_cam.point.z;
    double y_n = pt_cam.point.y / pt_cam.point.z;

    // 2. 直接使用 K 矩阵投影到像素坐标系
    double u = K(0,0) * x_n + K(0,2);  // u = fx * x_n + cx
    double v = K(1,1) * y_n + K(1,2);  // v = fy * y_n + cy

    // 3. 四舍五入到整数像素坐标
    return cv::Point2d(u,v);
}

void MultiSensorFusionPerceptionNodelet::printUwbPositions(
    const geometry_msgs::PointStamped& uwb_base,
    const geometry_msgs::PointStamped& uwb_odom,
    const geometry_msgs::PointStamped& uwb_cam,
    const cv::Point2d&                 uwb_uv)
{
    ROS_DEBUG("UWB in odom_est: x=%.2f, y=%.2f, z=%.2f",
            uwb_odom.point.x, uwb_odom.point.y, uwb_odom.point.z);
    ROS_DEBUG("UWB in base_link0: x=%.2f, y=%.2f, z=%.2f",
            uwb_base.point.x, uwb_base.point.y, uwb_base.point.z);
    ROS_DEBUG("UWB in zed2_left_camera_optical_frame: x=%.2f, y=%.2f, z=%.2f",
            uwb_cam.point.x, uwb_cam.point.y, uwb_cam.point.z);
    ROS_DEBUG("UWB in pixel: u=%.1d, v=%.1d",
            static_cast<int>(uwb_uv.x), static_cast<int>(uwb_uv.y));
}


// =========================================== Leg ================================================
void MultiSensorFusionPerceptionNodelet::printLegPositions(
    const geometry_msgs::PointStamped& leg_odom,
    const geometry_msgs::PointStamped& leg_base,
    const geometry_msgs::PointStamped& leg_cam,
    const cv::Point2d&                 leg_uv)
{
  ROS_DEBUG("Leg in odom_est:   x=%.2f, y=%.2f, z=%.2f",
            leg_odom.point.x, leg_odom.point.y, leg_odom.point.z);
  ROS_DEBUG("Leg in base_link0: x=%.2f, y=%.2f, z=%.2f",
            leg_base.point.x, leg_base.point.y, leg_base.point.z);
  ROS_DEBUG("Leg in zed2_left_camera_optical_frame: x=%.2f, y=%.2f, z=%.2f",
            leg_cam.point.x, leg_cam.point.y, leg_cam.point.z);
  ROS_DEBUG("Leg in pixel: u=%.1d, v=%.1d",
            static_cast<int>(leg_uv.x), static_cast<int>(leg_uv.y));
}



// =========================================== ZED ================================================
std::vector<Detection> MultiSensorFusionPerceptionNodelet::filterResult(const std::vector<Detection>& in){
    std::vector<Detection> out;
    for(const auto& t : in){
        if(t.classId != 0) continue;
        if (t.conf < yolo_conf_threshold_) continue;      // 只保留置信度 > 65%
        float w = t.bbox[2] - t.bbox[0];
        float h = t.bbox[3] - t.bbox[1];
        float r = h / w;
        if(r > 1.2f && r < 4.0f) out.push_back(t);
    }
    return out;
}

std::vector<DetectBox> MultiSensorFusionPerceptionNodelet::toDetectBoxPed(const std::vector<Detection>& src)
{
    std::vector<DetectBox> dst;
    dst.reserve(src.size());
    for (const auto& d : src)
        if (d.classId == 0)
            dst.emplace_back(DetectBox{d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3],
                           d.conf, static_cast<float>(d.classId)});
    return dst;
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



cv::Vec3f MultiSensorFusionPerceptionNodelet::pixel2camera(const cv::Point2d& uv, float z, const cv::Matx33d& K) {
    if (z <= 0.f) {
        ROS_WARN("深度 z=%.3f ≤ 0，返回零点！", z);
        return cv::Vec3f(0, 0, 0);
    }

    // 从内参矩阵 K 提取参数
    const double fx = K(0, 0);
    const double fy = K(1, 1);
    const double cx = K(0, 2);
    const double cy = K(1, 2);

    // 反投影计算（对称于 camera到pixel 的逆运算）
    const double x = (uv.x - cx) / fx * z;
    const double y = (uv.y - cy) / fy * z;

    return cv::Vec3f(x, y, z);
}


// 功能：在 YOLO 实例掩码内部取稳健中位深度（单位：米）
float MultiSensorFusionPerceptionNodelet::depthInsideMask(const std::vector<float>& maskMatrix,   // 单通道 float 掩码，长度 = 图像总像素
                      const cv::Mat& depthMap,                // 32FC1 深度图，单位毫米
                      const cv::Rect& bbox,                   // 检测框（已 clamp 在图内）
                      float thr)                              // 二值化阈值，默认 0.5f
{
    const int imgW = depthMap.cols;      // 整图宽度，用于计算一行偏移

    // 预分配容器，避免动态扩容；最坏情况框内全部有效
    std::vector<float> zVec;
    zVec.reserve(bbox.width * bbox.height);

    // 逐行扫描框内区域
    for (int dy = 0; dy < bbox.height; ++dy) {
        int y = bbox.y + dy;     // 当前行在整图中的 y 坐标

        // 指向本行第一个掩码值和深度值
        const float* m = maskMatrix.data() + y * imgW + bbox.x;  // 掩码第 y 行第 bbox.x 列
        const float* d = depthMap.ptr<float>(y) + bbox.x;        // 深度第 y 行第 bbox.x 列

        // 横向扫描框内宽度
        for (int dx = 0; dx < bbox.width; ++dx, ++m, ++d) {
            // 掩码>阈值 且 深度有效 → 收集
            if (*m > thr && *d > 0.f && *d < 15000.f)
                zVec.push_back(*d);          // 先存毫米，后面统一转米
        }
    }

    // 框内无有效像素 → 返回无效标志
    if (zVec.empty()) return -1.f;

    // 中位数 Robust 估计：nth_element 把中位值放到前半部分
    std::nth_element(zVec.begin(), zVec.begin() + zVec.size()/2, zVec.end());
    
    // 毫米 → 米 并返回
    return zVec[zVec.size()/2] * 0.001f;
}

MaskDepthStats MultiSensorFusionPerceptionNodelet::maskDepthStats(const std::vector<float>& maskMatrix,
                              const cv::Mat& depthMap,
                              const cv::Rect& bbox,
                              float thr)
{
    int maskPixels = 0;      // 掩码>阈值像素数
    int validPixels = 0;     // 掩码>阈值且深度有效像素数
    double sum_m = 0.0;      // 深度累加和（毫米）
    double sum_sq_m = 0.0;   // 深度平方累加和（毫米²）

    const int imgW = depthMap.cols;

    for (int dy = 0; dy < bbox.height; ++dy) {
        int y = bbox.y + dy;
        const float* m = maskMatrix.data() + y * imgW + bbox.x;
        const float* d = depthMap.ptr<float>(y) + bbox.x;

        for (int dx = 0; dx < bbox.width; ++dx, ++m, ++d) {
            if (*m > thr) {                       // ① 掩码内
                ++maskPixels;
                if (*d > 0.f && *d < 15000.f) {   // ② 深度有效
                    ++validPixels;
                    sum_m += *d;                  // 毫米
                    sum_sq_m += (*d) * (*d);      // 毫米²
                }
            }
        }
    }

    MaskDepthStats stats{};
    if (maskPixels == 0) {
        stats.validRatio = 0.f;
        stats.meanDepth  = 0.f;
        stats.varDepth   = 0.f;
        return stats;
    }

    stats.validRatio = static_cast<float>(validPixels) / maskPixels;

    double mean_mm = sum_m / validPixels;                 // 毫米均值
    double var_mm2 = (sum_sq_m / validPixels) - mean_mm * mean_mm; // 毫米方差

    stats.meanDepth = static_cast<float>(mean_mm * 0.001);      // mm → m
    stats.varDepth  = static_cast<float>(var_mm2 * 0.000001);   // mm² → m²
    return stats;
}

float MultiSensorFusionPerceptionNodelet::calcZedConfidence(
    float yolo_conf,
    float valid_ratio,
    float depth_var)
{
    // 深度方差 → 稳定性分数（var=0 时 1，var=0.1² 时 ≈0.37）
    float depth_stability = std::exp(-depth_var / 12.f);   // 可调 12.f
    // 三因子融合
    float conf = yolo_conf * valid_ratio * depth_stability;
    return std::clamp(conf, 0.f, 1.f);
}

void MultiSensorFusionPerceptionNodelet::printZedPositions(
    const geometry_msgs::PointStamped& zed_cam,
    const geometry_msgs::PointStamped& zed_base,
    const geometry_msgs::PointStamped& zed_odom,
    const cv::Point2d&                 zed_uv,
    int                                track_id)
{
    ROS_DEBUG("ZED[%d] in odom_est: x=%.2f, y=%.2f, z=%.2f",
            track_id, zed_odom.point.x, zed_odom.point.y, zed_odom.point.z);
    ROS_DEBUG("ZED[%d] in base_link0: x=%.2f, y=%.2f, z=%.2f",
            track_id, zed_base.point.x, zed_base.point.y, zed_base.point.z);
    ROS_DEBUG("ZED[%d] in zed2_left_camera_optical_frame: x=%.2f, y=%.2f, z=%.2f",
            track_id, zed_cam.point.x, zed_cam.point.y, zed_cam.point.z);   
    ROS_DEBUG("ZED[%d] in pixel: u=%.1d, v=%.1d",
            track_id, static_cast<int>(zed_uv.x), static_cast<int>(zed_uv.y));
}

// =========================================== fusion ================================================
void MultiSensorFusionPerceptionNodelet::pushSensorToSeq(
    const geometry_msgs::PointStamped& uwb,
    const geometry_msgs::PointStamped& leg,
    const geometry_msgs::PointStamped& zed,
    float uwb_conf, float leg_conf, float zed_conf)
{
    // 1. 定义归一化系数，与训练时保持一致（网络输入 0~1）
    constexpr float norm = 10.0f;
    seq_buf_.resize(12);          // 确保 12 元素
    // 2. 写入 UWB 的 xyz 并归一化，再写入置信度
    seq_buf_[0]  = uwb.point.x  / norm;
    seq_buf_[1]  = uwb.point.y  / norm;
    seq_buf_[2]  = uwb.point.z  / norm;
    seq_buf_[3]  = uwb_conf;
    // 3. 写入 Leg 的 xyz 并归一化，再写入置信度
    seq_buf_[4]  = leg.point.x  / norm;
    seq_buf_[5]  = leg.point.y  / norm;
    seq_buf_[6]  = leg.point.z  / norm;
    seq_buf_[7]  = leg_conf;
    // 4. 写入 ZED 的 xyz 并归一化，再写入置信度
    seq_buf_[8]  = zed.point.x  / norm;
    seq_buf_[9]  = zed.point.y  / norm;
    seq_buf_[10] = zed.point.z  / norm;
    seq_buf_[11] = zed_conf;

    // 在推入 TensorRT 前加一行
    printf("TRT-in: %.8f %.8f %.8f %.8f | %.8f %.8f %.8f %.8f | %.8f %.8f %.8f %.8f\n",
       seq_buf_[0], seq_buf_[1], seq_buf_[2], seq_buf_[3],
       seq_buf_[4], seq_buf_[5], seq_buf_[6], seq_buf_[7],
       seq_buf_[8], seq_buf_[9], seq_buf_[10], seq_buf_[11]);
    // // 2. 每 L 帧换一个新 seq 目录
    // const size_t L = 256;   // 一条样本的帧长，可调
    // if (g_frames_in_seq % L == 0) {
    //     g_seq_id++;      // 进新 seq
    // }

    // // 3. 两级目录：~/data/raw_npz/seqXXX/
    // const char* parent = "/home/zjn/temp/py_ws/sensor_fusion_nn/data/raw_npz";
    // mkdir(parent, 0755);
    // char seq_dir[512];
    // std::snprintf(seq_dir, sizeof(seq_dir), "%s/seq%llu", parent, g_seq_id - 1);
    // mkdir(seq_dir, 0755);

    // // 4. 写盘
    // char fname[512];
    // std::snprintf(fname, sizeof(fname), "%s/%06lu.npz",
    //               seq_dir, g_frames_in_seq % L);   // 每 seq 内从 000000 开始
    // cnpy::npz_save(fname, "x", seq_buf_.data(), {1, 12}, "w");
    // ROS_WARN("Wrote %s", fname);

    // g_frames_in_seq++;   // 总帧数 +1
}

void MultiSensorFusionPerceptionNodelet::doFusionInferAndPub()
{
    std::vector<float> out(4);
    if (fusion_->infer(seq_buf_, out))           // ← 这里失败就静默返回
    {
        geometry_msgs::PointStamped fused;
        fused.header.stamp    = ros::Time::now();
        fused.header.frame_id = "base_link0";
        fused.point.x = out[0] * 10.0f;   // 反归一化
        fused.point.y = out[1] * 10.0f;
        fused.point.z = out[2] * 10.0f;
        float fused_conf = out[3];
        
        ROS_INFO("融合后的坐标为：x=%2f, y=%2f, z=%2f, conf=%2f",fused.point.x,fused.point.y,fused.point.z,fused_conf);

        // 发布（先声明发布器 pub_fusion_ 即可）
        pub_fusion_.publish(fused);

        // 可视化 marker（可选）
        visualization_msgs::Marker m;
        m.header       = fused.header;
        m.ns           = "fused";
        m.id           = 0;
        m.type         = visualization_msgs::Marker::SPHERE;
        m.action       = visualization_msgs::Marker::ADD;
        m.pose.position = fused.point;
        m.pose.orientation.w = 1.0;
        m.scale.x = m.scale.y = m.scale.z = 0.25;
        m.color.r = 1.0; m.color.g = 1.0; m.color.b = 0.0; m.color.a = fused_conf;
        m.lifetime = ros::Duration(0.1);
        pub_fusion_marker_.publish(m);

        ROS_INFO("Fused  xyz=(%.2f %.2f %.2f) conf=%.2f", fused.point.x, fused.point.y, fused.point.z, fused_conf);
    }
    else
    {
        ROS_WARN("FusionNetTRT infer failed!");
    }
  
}

// =========================================== 工具函数 ================================================
void MultiSensorFusionPerceptionNodelet::print_confidences(float uwb_conf, float leg_conf, float zed_conf )
{
   ROS_DEBUG("UWB Confidence = %.2f", uwb_conf);
   ROS_DEBUG("Leg Confidence = %.2f", leg_conf);
   ROS_DEBUG("ZED Confidence = %.2f", zed_conf);
   
}

void MultiSensorFusionPerceptionNodelet::print_positions(
    const geometry_msgs::PointStamped& uwb_pt,
    const geometry_msgs::PointStamped& leg_pt,
    const geometry_msgs::PointStamped& zed_pt)
{
    ROS_DEBUG("UWB Position in base_link0: x=%.2f, y=%.2f, z=%.2f",
              uwb_pt.point.x, uwb_pt.point.y, uwb_pt.point.z);
    ROS_DEBUG("Leg Position in base_link0: x=%.2f, y=%.2f, z=%.2f",
              leg_pt.point.x, leg_pt.point.y, leg_pt.point.z);
    ROS_DEBUG("ZED Position in base_link0: x=%.2f, y=%.2f, z=%.2f",
              zed_pt.point.x, zed_pt.point.y, zed_pt.point.z);
    
}

void MultiSensorFusionPerceptionNodelet::print_transform_matrix(
    const std::string& from_frame,
    const std::string& to_frame)
{
    geometry_msgs::TransformStamped tf;
    try {
        tf = tf_buffer_.lookupTransform(to_frame, from_frame,
                                        ros::Time(0), ros::Duration(0.1));
    } catch (const tf2::TransformException& ex) {
        ROS_WARN_THROTTLE(1.0, "TF %s --> %s : %s",
                          from_frame.c_str(), to_frame.c_str(), ex.what());
        return;
    }

    Eigen::Isometry3d T = tf2::transformToEigen(tf);
    Eigen::Matrix4d   M = T.matrix();

    std::ostringstream oss;
    oss << "\n------ " << to_frame << " <-- " << from_frame << " ------\n";
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j)
            oss << std::setw(12) << std::fixed << std::setprecision(6)
                << M(i, j) << " ";
        oss << "\n";
    }
    oss << "----------------------------------------------";
    ROS_INFO_STREAM(oss.str());
}



PLUGINLIB_EXPORT_CLASS(multi_sensor_fusion::MultiSensorFusionPerceptionNodelet, nodelet::Nodelet)

} // namespace multi_sensor_fusion

