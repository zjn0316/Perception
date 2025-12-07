// v2,2025/12/7
#include "multi_sensor_fusion_perception_nodelet_v6.h"
#include <pluginlib/class_list_macros.h>

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
    deepsort_ = makeDeepSort(deepsort_engine_path_,64, 256, 0, deepsort_max_age_,deepsort_n_init_,deepsort_max_iou_,deepsort_max_cos_);

    sub_img_.reset(new message_filters::Subscriber<sensor_msgs::CompressedImage>(nh,img_topic_, 1));
    sub_depth_.reset(new message_filters::Subscriber<sensor_msgs::Image>(nh,depth_topic_, 1)); 
    pub_yolo_ = nh.advertise<sensor_msgs::Image>(vis_topic_, 1);
    pub_zed_ = nh.advertise<geometry_msgs::PointStamped>(zed_point_topic_, 1);
    pub_zed_marker_ = nh.advertise<visualization_msgs::MarkerArray>(zed_marker_topic_, 1);

    // leg
    sub_leg_.reset(new message_filters::Subscriber<people_tracker::PersonArray>(nh, leg_topic_, 2));
    pub_leg_ = nh.advertise<geometry_msgs::PointStamped>(leg_point_topic_, 1);
    pub_leg_marker_ = nh.advertise<visualization_msgs::MarkerArray>(leg_marker_topic_, 1);

    // spin
    // sync_.reset(new message_filters::Synchronizer<ApproxSyncPolicy>(ApproxSyncPolicy(50), *sub_img_, *sub_depth_, *sub_uwb_, *sub_leg_));
    // sync_->registerCallback(boost::bind(&MultiSensorFusionPerceptionNodelet::spin, this, _1, _2, _3,_4));
    sync_.reset(new message_filters::Synchronizer<ApproxSyncPolicy>(ApproxSyncPolicy(50), *sub_img_, *sub_depth_, *sub_uwb_));
    sync_->registerCallback(boost::bind(&MultiSensorFusionPerceptionNodelet::spin, this, _1, _2, _3));

    // 计时结束
    auto duration = (ros::Time::now() - start).toSec();
    ROS_INFO("onInit 花费 %.1f ms. Nodelet已成功初始化！", duration * 1000);
}




// void MultiSensorFusionPerceptionNodelet::spin(
//     const sensor_msgs::CompressedImageConstPtr& img_msg,
//     const sensor_msgs::ImageConstPtr& depth_msg,
//     const nlink_parser::LinktrackAoaNodeframe0::ConstPtr& uwb_msg,
//     const people_tracker::PersonArray::ConstPtr& leg_msg)
void MultiSensorFusionPerceptionNodelet::spin(
    const sensor_msgs::CompressedImageConstPtr& img_msg,
    const sensor_msgs::ImageConstPtr& depth_msg,
    const nlink_parser::LinktrackAoaNodeframe0::ConstPtr& uwb_msg)
{
    auto start = ros::Time::now();
    ros::Time t0, t1, t2, t3, t4, t5, t6, t8, t9;   // 全部提前声明
    t0 = ros::Time::now();
    lookupStaticTransforms();
    t1 = ros::Time::now();
    // debug 打印变换矩阵示例：
    // print_transform_matrix("nlink", "zed2_left_camera_optical_frame");


    // 处理各个传感器回调
    /* ======================== 1. UWB 处理 ========================== */
    {
        /* -------------- 1. 有效性检查 -------------- */
        if (uwb_msg->nodes.empty() || uwb_msg->nodes.at(0).role != 1)
            goto uwb_done;

        /* -------------- 2. 极坐标 → nlink直角坐标 -------------- */
        // 计算距离和角度
        float dis = uwb_msg->nodes.at(0).dis;// 距离
        float ang = uwb_msg->nodes.at(0).angle * M_PI / 180.0f;// 角度
        // 计算uwb在nlink坐标系下的坐标
        geometry_msgs::PointStamped uwb_pt_nlink;// uwb在nlink坐标系下的坐标
        uwb_pt_nlink.header.stamp    = uwb_msg->header.stamp;
        uwb_pt_nlink.header.frame_id = "nlink";
        uwb_pt_nlink.point.x = dis * std::cos(ang);
        uwb_pt_nlink.point.y = dis * std::sin(ang);
        uwb_pt_nlink.point.z = 0.45;// nlink坐标系下uwb标签大约高度 45 m
        // ROS_DEBUG("UWB in nlink: x=%.2f, y=%.2f, z=%.2f",uwb_pt_nlink.point.x, uwb_pt_nlink.point.y, uwb_pt_nlink.point.z);

        /* -------------- 3. nlink坐标 → base坐标-------------- */
        // 计算uwb在base_link0坐标系下的坐标
        geometry_msgs::PointStamped uwb_pt_base;// uwb在base_link0坐标系下的坐标
        tf2::doTransform(uwb_pt_nlink, uwb_pt_base, tf_nlink2base_);
        ROS_DEBUG("UWB in base_link0: x=%.2f, y=%.2f, z=%.2f",uwb_pt_base.point.x, uwb_pt_base.point.y, uwb_pt_base.point.z);

        /* -------------- 4. nlink坐标 → odom_est坐标-------------- */
        // 计算uwb在odom_est坐标系下的坐标
        geometry_msgs::PointStamped uwb_pt_odom;// uwb在odom_est坐标系下的坐标
        tf2::doTransform(uwb_pt_nlink, uwb_pt_odom, tf_base2odom_);  
        // ROS_DEBUG("UWB in odom_est: x=%.2f, y=%.2f, z=%.2f",uwb_pt_odom.point.x, uwb_pt_odom.point.y, uwb_pt_odom.point.z);
        
        /* -------------- 5. nlink坐标 → 相机坐标-------------- */
        geometry_msgs::PointStamped uwb_pt_cam;// uwb在相机坐标系下的坐标
        tf2::doTransform(uwb_pt_nlink, uwb_pt_cam, tf_nlink2cam_);
        // ROS_DEBUG("UWB in camear: x=%.2f, y=%.2f, z=%.2f", uwb_pt_cam.point.x, uwb_pt_cam.point.y, uwb_pt_cam.point.z);

        /* -------------- 6. 相机坐标 → 像素坐标-------------- */
        uwb_uv_ = camera2pixel(uwb_pt_cam, K);  
        ROS_DEBUG("UWB in pixel: u=%.1d, v=%.1d", static_cast<int>(uwb_uv_.x),static_cast<int>(uwb_uv_.y));

        /* -------------- 8. 发布话题 -------------- */
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
        marker.color.a = 1.0f;
        marker.lifetime = ros::Duration(1.0);// 持续 1 秒
        pub_uwb_marker_.publish(marker);
    }
    t2 = ros::Time::now();
uwb_done:;

    /* ======================== 2. ZED 处理 ========================== */
    {
        /* -------------- 1. 解压彩色图 2ms-------------- */
        cv_bridge::CvImagePtr cv_ptr_zed;
        try {
            cv_ptr_zed = cv_bridge::toCvCopy(img_msg, "bgr8");
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            goto zed_done;
        }
        cv::Mat img_zed = cv_ptr_zed->image;
        t3 = ros::Time::now();

        /* -------------- 2. YOLO 检测 + 过滤 10ms -------------- */
        auto y0 = ros::Time::now();
        std::vector<Detection> res    = yolo_->inference(img_zed);
        std::vector<Detection> res_fd = filterResult(res);
        auto y1 = ros::Time::now();
        ROS_INFO("YOLO cost %.1f ms", (y1 - y0).toSec() * 1000.0);
        t4 = ros::Time::now();

        /* -------------- 3. DeepSorT 10 ms-------------- */
        t5 = ros::Time::now();                 // <-- 新增

        std::vector<DetectBox> dets_deepsort;
        for (const auto& d : res_fd) {
            if (d.classId != 0) continue;          // 只要行人
            DetectBox box;
            box.x1 = d.bbox[0];
            box.y1 = d.bbox[1];
            box.x2 = d.bbox[2];
            box.y2 = d.bbox[3];
            box.confidence = d.conf;
            box.classID    = d.classId;
            dets_deepsort.emplace_back(box);
        }
        printf("Number of detections: %zu\n", dets_deepsort.size());
        deepsort_->sort(img_zed, dets_deepsort);   // 内部已做特征提取+数据关联

        t6 = ros::Time::now();                 // <-- 新增
        ROS_INFO("Deepsort cost %.1f ms", (t6 - t5).toSec() * 1000.0);  // <-- 新增

        /* -------------- 4. 深度图 → cv::Mat (32FC1) 0ms -------------- */
        cv::Mat depth_zed;
        try {
            depth_zed = cv_bridge::toCvShare(depth_msg,sensor_msgs::image_encodings::TYPE_32FC1)->image;
        } catch (cv_bridge::Exception& e) {
            NODELET_ERROR("depth cv_bridge exception: %s", e.what());
            goto zed_done;
        }

        /* -------------- 5.像素坐标系投影到图像坐标系 2-D 框中心 → 3-D 坐标 -------------- */
        visualization_msgs::MarkerArray marker_array_zed;
        for (const auto& d : dets_deepsort) {
            if (d.classID != 0) continue;// 只处理行人
            int zed_u = static_cast<int>((d.x1 + d.x2) * 0.5f);
            int zed_v = static_cast<int>((d.y1 + d.y2) * 0.5f);
            // 计算图像中心
            zed_u = std::max(0, std::min(zed_u, depth_zed.cols - 1));
            zed_v = std::max(0, std::min(zed_v, depth_zed.rows - 1));
            // 根据像素索引得到深度值
            float depth_raw = median3x3(depth_zed, zed_u, zed_v);
            if (depth_raw >= 15000 || depth_raw <= 0.f) continue;
            float depth = depth_raw * 0.001f;
            // ROS_DEBUG("depth = %.2f m",depth);
            // 像素 → 图像坐标系
            cv::Point2d uv(zed_u, zed_v);
            cv::Vec3f xyz = pixel2camera(uv, depth, K);
            // ROS_DEBUG("zed in camera x=%.1f, y=%.1f,z=%.1f", xyz[0], xyz[1],xyz[2]);
            // 相机坐标系下坐标 
            geometry_msgs::PointStamped zed_pt_cam;
            zed_pt_cam.header.stamp    = depth_msg->header.stamp;
            zed_pt_cam.header.frame_id = "zed2_left_camera_optical_frame";
            zed_pt_cam.point.x         = xyz[0];
            zed_pt_cam.point.y         = xyz[1];
            zed_pt_cam.point.z         = xyz[2];
            //  base_link0下坐标
            geometry_msgs::PointStamped zed_pt_base;
            zed_pt_base.header.stamp    = zed_pt_cam.header.stamp; 
            tf2::doTransform(zed_pt_cam, zed_pt_base, tf_zed2base_);  // 用成员变量
            ROS_DEBUG("ZED in base_link0: x=%.2f, y=%.2f, z=%.2f",zed_pt_base.point.x, zed_pt_base.point.y, zed_pt_base.point.z);
            geometry_msgs::PointStamped zed_pt_odom;
            zed_pt_odom.header.stamp    = zed_pt_cam.header.stamp; 
            tf2::doTransform(zed_pt_cam, zed_pt_odom, tf_zed2odom_);  // 用成员变量
            // ROS_DEBUG("ZED in odom_est: x=%.2f, y=%.2f, z=%.2f",zed_pt_odom.point.x, zed_pt_odom.point.y, zed_pt_odom.point.z);

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
            marker.color.a = 1.0f;
            marker.lifetime = ros::Duration(1.0);
            marker_array_zed.markers.push_back(marker);
        }
        pub_zed_marker_.publish(marker_array_zed);


        /* -------------- 6. 可视化发布图片 -------------- */
        cv::Mat& img_seg = cv_ptr_zed->image;
        // 画框图
        YoloDetector::draw_image(img_seg, res_fd);
        // 画 zed 红点
        for (size_t i = 0; i < res_fd.size(); ++i) {
            const auto& d = res_fd[i];
            if (d.classId != 0) continue;
            int zed_u = static_cast<int>((d.bbox[0] + d.bbox[2]) * 0.5f);
            int zed_v = static_cast<int>((d.bbox[1] + d.bbox[3]) * 0.5f);
            ROS_DEBUG("ZED in pixel: u=%.1d, v=%.1d", zed_u, zed_v);
            /* 画跟踪框 + ID */
            cv::Scalar color = cv::Scalar(0, 255, 0);                  // 可改成根据 ID 选颜色
            cv::rectangle(img_seg,cv::Point(d.bbox[0], d.bbox[1]),cv::Point(d.bbox[2], d.bbox[3]),color, 2);
            cv::putText(img_seg,
                        "ID:" + std::to_string((int)dets_deepsort[i].trackID),
                        cv::Point(d.bbox[0], d.bbox[1] - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);

            /* 中心红点照旧 */
            cv::circle(img_seg, cv::Point(zed_u, zed_v), 4,cv::Scalar(0, 0, 255), -1);
        }
        
        // 画 UWB 蓝点
        cv::circle(img_seg,cv::Point(static_cast<int>(uwb_uv_.x),static_cast<int>(uwb_uv_.y)),4, cv::Scalar(255, 0, 0), -1);

        sensor_msgs::ImagePtr msg_seg =cv_bridge::CvImage(img_msg->header, "bgr8", img_seg).toImageMsg();
        pub_yolo_.publish(msg_seg);
    }
    t8 = ros::Time::now();
zed_done:;
    // /* ---------- 3. Leg 处理（原 legCallback）---------- */
    // {
    //     visualization_msgs::MarkerArray ma_leg;
    //     ma_leg.markers.reserve(leg_msg->people.size());
    //     for (size_t i = 0; i < leg_msg->people.size(); ++i) {
    //         visualization_msgs::Marker m;
    //         m.header       = leg_msg->header;
    //         m.ns           = "people";
    //         m.id           = static_cast<int>(i);
    //         m.type         = visualization_msgs::Marker::SPHERE;
    //         m.action       = visualization_msgs::Marker::ADD;
    //         m.pose.position = leg_msg->people[i].pose.position;
    //         m.pose.position.z = 0.3;
    //         m.pose.orientation.w = 1.0;
    //         m.scale.x = m.scale.y = m.scale.z = 0.2;
    //         m.color.r = 0.0;
    //         m.color.g = 1.0;
    //         m.color.b = 0.0;
    //         m.color.a = 0.9;
    //         m.lifetime = ros::Duration(1);
    //         ma_leg.markers.push_back(m);
    //     }
    //     pub_leg_marker_.publish(ma_leg);
    // }

    /* ===================== 打印分段耗时 ===================== */
    ROS_INFO("TF-lookup=%.1f  UWB=%.1f  YOLO=%.f  DeepSORT=%.1f  "
             "Depth+Marker+Pub=%.1f ms",
             (t1 - t0).toSec() * 1000,      // 纯 TF lookup
             (t2 - t1).toSec() * 1000,      // UWB 段
             (t4 - t3).toSec() * 1000,      // YOLO 检测
             (t6 - t5).toSec() * 1000,      // DeepSORT（含特征）
             (t8 - t6).toSec() * 1000);     // 深度读取+Marker+图像发布

    auto duration = (ros::Time::now() - start).toSec();
    ROS_INFO("spin 花费 %.1f ms.------------spin finished------------.", duration * 1000);
}

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
    }
    catch (tf2::TransformException& ex)
    {
        ROS_WARN_THROTTLE(1.0, "[lookupStaticTransforms] %s", ex.what());
    }
}



/*----------------------工具函数--------------------------*/
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
    // TODO: 可以添加更多参数
    // target_person(target_person_id,target_person_position,target_person_conf)
    // other_person(other_person_id,other_person_position,other_person_conf)

    // 路径参数
    pnh.param("yolo_engine_path", yolo_engine_path_,std::string("/home/zjn/catkin_ws/src/multi_sensor_fusion_perception_nodelet/engine/yolo11s-seg.plan"));
    pnh.param("deepsort_engine_path", deepsort_engine_path_,std::string("/home/zjn/catkin_ws/src/multi_sensor_fusion_perception_nodelet/engine/deepsort.engine"));

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
    ROS_INFO("============== 路径 =============");
    ROS_INFO("  YOLO Engine Path: %s", yolo_engine_path_.c_str());
    ROS_INFO("  DeepSort Engine Path: %s", deepsort_engine_path_.c_str());
    ROS_INFO("============ 其他参数 ============");
    ROS_INFO("  Yolo Conf Thresh: %.2f", yolo_conf_threshold_);
    ROS_INFO("  DeepSort Max Age: %d",   deepsort_max_age_);
    ROS_INFO("  DeepSort N Init:  %d",   deepsort_n_init_);
    ROS_INFO("  DeepSort Max IoU: %.2f", deepsort_max_iou_);
    ROS_INFO("  DeepSort Max Cos: %.2f", deepsort_max_cos_);
    ROS_INFO("================================");
}


PLUGINLIB_EXPORT_CLASS(multi_sensor_fusion::MultiSensorFusionPerceptionNodelet, nodelet::Nodelet)

} // namespace multi_sensor_fusion

