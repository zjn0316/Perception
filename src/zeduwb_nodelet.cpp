#include "zeduwb_nodelet.h"
#include <chrono> // [新增] 用于计算时间

namespace zeduwb_ns {

void ZedUwbNodelet::onInit() {
    nh_ = getNodeHandle();
    private_nh_ = getPrivateNodeHandle();

    ROS_INFO("Initializing ZedUwbNodelet...");

    // 1. 初始化 TF
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(tf_buffer_);

    // 2. 加载参数
    private_nh_.param("yolo_engine_path", yolo_engine_path_, std::string("/home/zjn/catkin_ws/src/multi_sensor_fusion_perception_nodelet/engine/yolo11n.engine"));

    // 3. 初始化 YOLO
    ROS_INFO("Loading YOLO from: %s", yolo_engine_path_.c_str());
    yolo_.reset(new YoloDetector(yolo_engine_path_));

    // 4. 相机内参 (建议改为从 CameraInfo 话题读取，这里暂用硬编码)
    K_ = (cv::Mat_<double>(3, 3) << 
          267.4111022949219, 0.0, 312.1522827148438, 
          0.0, 267.4111022949219, 184.7516326904297, 
          0.0, 0.0, 1.0);

    // 5. 订阅话题
    img_sub_   = std::make_shared<message_filters::Subscriber<sensor_msgs::Image>>(nh_, "/zed2/zed_node/rgb/image_rect_color", 1);
    depth_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::Image>>(nh_, "/zed2/zed_node/depth/depth_registered", 1);
    uwb_sub_   = std::make_shared<message_filters::Subscriber<nlink_parser::LinktrackAoaNodeframe0>>(nh_, "/uwb_filter_polar", 1);

    // 6. 同步器
    sync_ = std::make_shared<message_filters::Synchronizer<ApproxSyncPolicy>>(ApproxSyncPolicy(50), *img_sub_, *depth_sub_, *uwb_sub_);
    sync_->registerCallback(boost::bind(&ZedUwbNodelet::syncCallback, this, _1, _2, _3));

    // 7. 初始化发布者
    pub_image_compressed_ = nh_.advertise<sensor_msgs::CompressedImage>("/zed_uwb/synced_image/compressed", 1);
    pub_uwb_3d_ = nh_.advertise<geometry_msgs::PointStamped>("/uwb/base/3d_point", 1);
    pub_uwb_2d_ = nh_.advertise<geometry_msgs::PointStamped>("/uwb/pixel/2d_point", 1);
    pub_uwb_marker_ = nh_.advertise<visualization_msgs::Marker>("/uwb/marker/3d_point", 1);
    pub_bbox_   = nh_.advertise<std_msgs::Float32MultiArray>("/ostrack_input_bbox", 1);

    ROS_INFO("ZedUwbNodelet Init Done.");
}

// === 核心回调函数 ===
void ZedUwbNodelet::syncCallback(const sensor_msgs::ImageConstPtr& img_msg,
                                 const sensor_msgs::ImageConstPtr& depth_msg,
                                 const nlink_parser::LinktrackAoaNodeframe0ConstPtr& uwb_msg) {
    // 1. 转 OpenCV 格式
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    cv::Mat img = cv_ptr->image;



    // ================== UWB 处理 (核心修改) ==================
    // 检查数据有效性
    if (uwb_msg->nodes.empty()) return;

    // [新增] 解析 UWB 数据 (极坐标 -> 直角坐标)
    // 你的参考代码 v10.cpp 里是这么写的：
    float dis = uwb_msg->nodes.at(0).dis;
    float ang = uwb_msg->nodes.at(0).angle * M_PI / 180.0f; // 角度转弧度

    // 构造一个临时的 PointStamped 用于 TF 转换
    geometry_msgs::PointStamped uwb_pt_raw;
    uwb_pt_raw.header = uwb_msg->header;
    uwb_pt_raw.header.frame_id = "nlink"; // 原始数据通常在 nlink 坐标系
    uwb_pt_raw.point.x = dis * std::cos(ang);
    uwb_pt_raw.point.y = dis * std::sin(ang);
    uwb_pt_raw.point.z = 0.54; // 假设高度

    // ---------------- 下面是坐标转换逻辑 (和之前一样，只是输入变了) ----------------
    geometry_msgs::PointStamped uwb_base_pt;
    geometry_msgs::PointStamped uwb_cam_pt;
    cv::Point uwb_pixel_pt(-1, -1);
    bool uwb_valid = false;

    try {
        // A. 转到 base_link (用于 3D 距离)
        if (tf_buffer_.canTransform("base_link0", uwb_msg->header.frame_id, ros::Time(0))) {
            tf_buffer_.transform(uwb_pt_raw, uwb_base_pt, "base_link0"); // 注意这里传入的是 uwb_pt_raw
            pub_uwb_3d_.publish(uwb_base_pt);

            // 发布 UWB Marker
            visualization_msgs::Marker marker;
            marker.header = uwb_base_pt.header;
            marker.ns = "uwb";
            marker.id = 0;
            marker.type = visualization_msgs::Marker::SPHERE;
            marker.action = visualization_msgs::Marker::ADD;
            marker.pose.position = uwb_base_pt.point;
            marker.pose.orientation.w = 1.0;
            marker.scale.x = marker.scale.y = marker.scale.z = 0.2;
            marker.color.b = 1.0f; marker.color.a = 1.0f; // 蓝色
            pub_uwb_marker_.publish(marker);
        }

        // B. 转到 相机光心 (用于 2D 投影)
        std::string cam_frame = "zed2_left_camera_optical_frame";
        if (tf_buffer_.canTransform(cam_frame, uwb_msg->header.frame_id, ros::Time(0))) {
            tf_buffer_.transform(uwb_pt_raw, uwb_cam_pt, cam_frame); // 注意这里传入的是 uwb_pt_raw
            
            double x = uwb_cam_pt.point.x;
            double y = uwb_cam_pt.point.y;
            double z = uwb_cam_pt.point.z;

            // ... (投影逻辑保持不变) ...
             if (z > 0.1) {
                int u = static_cast<int>(K_.at<double>(0, 0) * x / z + K_.at<double>(0, 2));
                int v = static_cast<int>(K_.at<double>(1, 1) * y / z + K_.at<double>(1, 2));
                // ...
                if (u >= 0 && u < img.cols && v >= 0 && v < img.rows) {
                    uwb_pixel_pt = cv::Point(u, v);
                    uwb_valid = true;

                    // 发布 2D 坐标
                    geometry_msgs::PointStamped pixel_msg;
                    pixel_msg.header = img_msg->header;
                    pixel_msg.point.x = u;
                    pixel_msg.point.y = v;
                    pixel_msg.point.z = z;
                    pub_uwb_2d_.publish(pixel_msg);

                    // 画 UWB 蓝点
                    cv::circle(img, uwb_pixel_pt, 8, cv::Scalar(255, 0, 0), -1);
                    cv::putText(img, "UWB", cv::Point(u+10, v), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
                }
            }
        }
    } catch (tf2::TransformException &ex) {
        ROS_WARN_THROTTLE(1.0, "TF Error: %s", ex.what());
    }


    // ================== YOLO 处理 ==================
    // [修改] 1. 开始计时
    auto t_start = std::chrono::high_resolution_clock::now();
    
    // 执行推理
    std::vector<Detection> res = yolo_->inference(img);
    
    // [修改] 2. 结束计时
    auto t_end = std::chrono::high_resolution_clock::now();

    // [修改] 3. 计算耗时和 FPS
    // 计算毫秒耗时
    double infer_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    // 计算 FPS (1000ms / 耗时)
    double infer_fps = (infer_time_ms > 0) ? (1000.0 / infer_time_ms) : 0.0;

    // 2. 过滤
    std::vector<Detection> res_fd = FilterResult(res);

    // 3. 画框 (新库 bbox 是 x1, y1, x2, y2)
    for (const auto& d : res_fd) {
        cv::rectangle(img, cv::Point(d.bbox[0], d.bbox[1]), cv::Point(d.bbox[2], d.bbox[3]), cv::Scalar(0, 255, 0), 2);
    }

    // [修改] 4. 将推理 FPS 画在图像左上角
    // 格式例如: "YOLO: 15ms (66 FPS)"
    std::string stats_text = "YOLO: " + std::to_string(static_cast<int>(infer_time_ms)) + "ms (" + 
                             std::to_string(static_cast<int>(infer_fps)) + " FPS)";
                             
    cv::putText(img, stats_text, cv::Point(20, 40), 
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2); // 黄色字体


                


    // ================== 匹配逻辑 (UWB <-> YOLO) ==================
    int best_idx = -1;
    double min_dist_sq = 100000.0;

    if (uwb_valid && !res_fd.empty()) {
        for (size_t i = 0; i < res_fd.size(); ++i) {
            // [修改] 计算中心点：(x1+x2)/2, (y1+y2)/2
            float cx = (res_fd[i].bbox[0] + res_fd[i].bbox[2]) / 2.0f;
            float cy = (res_fd[i].bbox[1] + res_fd[i].bbox[3]) / 2.0f;

            float dx = cx - uwb_pixel_pt.x;
            float dy = cy - uwb_pixel_pt.y;
            float dist_sq = dx*dx + dy*dy;

            // 距离阈值 (例如 150像素)
            if (dist_sq < 150.0 * 150.0 && dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                best_idx = static_cast<int>(i);
            }
        }
    }

    // ================== 发布结果 ==================
    if (best_idx != -1) {
        const auto& d = res_fd[best_idx];
        
        // 画选中框 (红色)
        cv::rectangle(img, cv::Point(d.bbox[0], d.bbox[1]), cv::Point(d.bbox[2], d.bbox[3]), cv::Scalar(0, 0, 255), 3);
        
        // 发布 BBox [x, y, w, h] 给 OSTrack
        std_msgs::Float32MultiArray box_msg;
        float w = d.bbox[2] - d.bbox[0]; // x2 - x1
        float h = d.bbox[3] - d.bbox[1]; // y2 - y1
        
        box_msg.data.push_back(d.bbox[0]); // x (left)
        box_msg.data.push_back(d.bbox[1]); // y (top)
        box_msg.data.push_back(w);
        box_msg.data.push_back(h);
        pub_bbox_.publish(box_msg);
    }

    // ================== 发布压缩图 ==================
    std::vector<uchar> buf;
    if (cv::imencode(".jpg", img, buf)) {
        sensor_msgs::CompressedImage compressed_msg;
        compressed_msg.header = img_msg->header;
        compressed_msg.format = "jpeg";
        compressed_msg.data = buf;
        pub_image_compressed_.publish(compressed_msg);
    }
}

std::vector<Detection> ZedUwbNodelet::FilterResult(const std::vector<Detection>& in) {
    std::vector<Detection> out;
    for(const auto& t : in){
        // [修改] 字段名 classId
        if(t.classId != 0) continue; 
        
        // [修改] 宽高计算 (x2-x1, y2-y1)
        float w = t.bbox[2] - t.bbox[0];
        float h = t.bbox[3] - t.bbox[1];
        
        if (w > 0 && h > 0) {
            float r = h / w;
            if(r > 1.0f && r < 4.5f) out.push_back(t);
        }
    }
    return out;
}


} // namespace zeduwb_ns

// 注册 Nodelet
PLUGINLIB_EXPORT_CLASS(zeduwb_ns::ZedUwbNodelet, nodelet::Nodelet)