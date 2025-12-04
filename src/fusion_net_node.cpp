#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <fstream>  // Added for file operations

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            ROS_WARN("TensorRT: %s", msg);
        } else {
            ROS_INFO("TensorRT: %s", msg);
        }
    }
} gLogger;

class FusionNetNode {
public:
    FusionNetNode() : engine_(nullptr), context_(nullptr), runtime_(nullptr),
                     input_d_(nullptr), output_d_(nullptr), input_h_(nullptr), output_h_(nullptr) {
        ros::NodeHandle nh;
        ros::NodeHandle private_nh("~");

        // Get parameters
        std::string engine_path;
        if (!private_nh.getParam("engine_path", engine_path)) {
            ROS_ERROR("Failed to get engine_path parameter");
            throw std::runtime_error("Missing engine_path parameter");
        }

        // Load TensorRT engine
        if (!loadEngine(engine_path)) {
            ROS_ERROR("Failed to load TensorRT engine");
            throw std::runtime_error("Failed to load engine");
        }
            // ================== 1. 加载后立刻打印 ==================
        ROS_INFO("=== TensorRT Engine Summary ===");
        ROS_INFO("Engine file : %s", engine_path.c_str());
        ROS_INFO("Number of bindings: %d", engine_->getNbBindings());
        for (int i = 0; i < engine_->getNbBindings(); ++i) {
            const char* name = engine_->getBindingName(i);
            bool isInput = engine_->bindingIsInput(i);
            auto dims = engine_->getBindingDimensions(i);
            std::string shape;
            for (int j = 0; j < dims.nbDims; ++j)
                shape += std::to_string(dims.d[j]) + (j == dims.nbDims - 1 ? "" : "×");
            ROS_INFO("  Binding[%d] %s  %s  shape=%s", i, isInput ? "INPUT" : "OUTPUT", name, shape.c_str());
        }
        ROS_INFO("================================");

        // Setup input/output buffers
        setupBuffers();

        // Setup ROS subscribers and publishers
        sub_ = nh.subscribe("input_confidences", 10, &FusionNetNode::confidenceCallback, this);
        pub_ = nh.advertise<std_msgs::Float32MultiArray>("fusion_weights", 10);

            // 测试定时器：启动后 0.2 s 开始，每 2 s 跑一轮
        test_timer_ = nh.createTimer(ros::Duration(2.0), &FusionNetNode::testTimerCallback, this);

        ROS_INFO("FusionNet node initialized");
    }

    ~FusionNetNode() {
        // Cleanup
        if (context_) {
            delete context_;
        }
        if (engine_) {
            delete engine_;
        }
        if (runtime_) {
            delete runtime_;
        }

        if (input_d_) {
            cudaFree(input_d_);
        }
        if (output_d_) {
            cudaFree(output_d_);
        }
        if (input_h_) {
            delete[] input_h_;
        }
        if (output_h_) {
            delete[] output_h_;
        }
    }

private:
    bool loadEngine(const std::string& engine_path) {
        std::ifstream engine_file(engine_path, std::ios::binary);
        if (!engine_file) {
            ROS_ERROR("Failed to open engine file: %s", engine_path.c_str());
            return false;
        }

        engine_file.seekg(0, std::ios::end);
        size_t size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);
        std::vector<char> engine_data(size);
        engine_file.read(engine_data.data(), size);
        engine_file.close();

        runtime_ = nvinfer1::createInferRuntime(gLogger);
        if (!runtime_) {
            ROS_ERROR("Failed to create TensorRT runtime");
            return false;
        }

        engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
        if (!engine_) {
            ROS_ERROR("Failed to deserialize engine");
            return false;
        }

        context_ = engine_->createExecutionContext();
        if (!context_) {
            ROS_ERROR("Failed to create execution context");
            return false;
        }

        return true;
    }

    void setupBuffers() {
    nvinfer1::Dims input_dims = engine_->getBindingDimensions(0);
    nvinfer1::Dims output_dims = engine_->getBindingDimensions(1);

    size_t input_count = 1, output_count = 1;
    for (int i = 0; i < input_dims.nbDims; ++i) input_count *= input_dims.d[i];
    for (int i = 0; i < output_dims.nbDims; ++i) output_count *= output_dims.d[i];

    size_t input_bytes = input_count * sizeof(float);
    size_t output_bytes = output_count * sizeof(float);

    ROS_INFO("Buffer setup: input_count=%zu  output_count=%zu", input_count, output_count);
    ROS_INFO("CUDA malloc:  input=%.1f MB  output=%.1f MB",
             input_bytes / 1048576.f, output_bytes / 1048576.f);

    cudaError_t err;
    err = cudaMalloc((void**)&input_d_, input_bytes);
    if (err != cudaSuccess) throw std::runtime_error("CUDA input malloc failed");
    err = cudaMalloc((void**)&output_d_, output_bytes);
    if (err != cudaSuccess) throw std::runtime_error("CUDA output malloc failed");

    input_h_ = new float[input_count];
    output_h_ = new float[output_count];

    ROS_INFO("Buffers ready - device: input_d=%p output_d=%p | host: input_h=%p output_h=%p",
             input_d_, output_d_, input_h_, output_h_);
}

    void confidenceCallback(const std_msgs::Float32MultiArray::ConstPtr& msg) {
        // Check input size matches expected
        size_t expected_input_size = 1 * 20 * 3;  // Adjust based on your model
        if (msg->data.size() != expected_input_size) {
            ROS_WARN("Received input with size %zu, expected %zu", 
                    msg->data.size(), expected_input_size);
            return;
        }

        // Copy input data
        std::copy(msg->data.begin(), msg->data.end(), input_h_);

        // Copy input to device
        cudaMemcpy(input_d_, input_h_, msg->data.size() * sizeof(float), 
                  cudaMemcpyHostToDevice);

        // Run inference
        void* bindings[] = {input_d_, output_d_};
        if (!context_->executeV2(bindings)) {
            ROS_ERROR("Failed to execute TensorRT inference");
            return;
        }

        // Copy output back to host
        size_t output_size = 3;  // Adjust based on your model
        cudaMemcpy(output_h_, output_d_, output_size * sizeof(float), 
                  cudaMemcpyDeviceToHost);

        // Publish results
        std_msgs::Float32MultiArray weights_msg;
        weights_msg.data.assign(output_h_, output_h_ + output_size);
        pub_.publish(weights_msg);
    }

        // 测试用定时器回调
    void testTimerCallback(const ros::TimerEvent&)
    {
        // 3 组假数据：高视觉、高雷达、高 UWB
        std::vector<std::vector<float>> tests = {
            {0.9f,0.9f,0.9f, 0.9f,0.9f,0.9f, 0.9f,0.9f,0.9f, 0.9f,0.9f,0.9f, 0.9f,0.9f,0.9f, 0.9f,0.9f,0.9f, 0.9f,0.9f,0.9f},   // 视觉强
            {0.3f,0.3f,0.3f, 0.3f,0.3f,0.3f, 0.3f,0.3f,0.3f, 0.3f,0.3f,0.3f, 0.3f,0.3f,0.3f, 0.3f,0.3f,0.3f, 0.3f,0.3f,0.3f},   // 视觉弱
            {0.5f,0.5f,0.5f, 0.5f,0.5f,0.5f, 0.5f,0.5f,0.5f, 0.5f,0.5f,0.5f, 0.5f,0.5f,0.5f, 0.5f,0.5f,0.5f, 0.5f,0.5f,0.5f}    // 中性
        };

        for (const auto& in : tests) {
            // 直接拷到 host → device → infer
            std::copy(in.begin(), in.end(), input_h_);
            cudaMemcpy(input_d_, input_h_, in.size() * sizeof(float), cudaMemcpyHostToDevice);
            void* bindings[] = {input_d_, output_d_};
            if (!context_->executeV2(bindings)) {
                ROS_ERROR("Test inference failed");
                continue;
            }
            cudaMemcpy(output_h_, output_d_, 3 * sizeof(float), cudaMemcpyDeviceToHost);

            // 终端可视化
            ROS_INFO("TEST IN = [%6.3f ... %6.3f]  ->  OUT = [w_v=%6.3f  w_l=%6.3f  w_u=%6.3f]",
                     in[0], in.back(), output_h_[0], output_h_[1], output_h_[2]);

            // 同时发布，方便 rostopic echo 看
            std_msgs::Float32MultiArray msg;
            msg.data.assign(output_h_, output_h_ + 3);
            pub_.publish(msg);
            ros::Duration(0.5).sleep();   // 每 0.5 s 一组
        }
        ROS_INFO("===== Test round finished =====");
    }

    // TensorRT members
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;

    // CUDA buffers
    float* input_d_;
    float* output_d_;
    float* input_h_;
    float* output_h_;

    // ROS members
    ros::Subscriber sub_;
    ros::Publisher pub_;
    ros::Timer test_timer_;   // 测试定时器
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "fusion_net_node");
    try {
        FusionNetNode node;
        ros::spin();
    } catch (const std::exception& e) {
        ROS_ERROR("%s", e.what());
        return 1;
    }
    return 0;
}
