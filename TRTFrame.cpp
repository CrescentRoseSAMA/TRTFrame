#include "TRTFrame.hpp"
#include <opencv4/opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <NvOnnxParser.h>
#include <logger.h>
#include <fstream>
#include <cuda.h>
#include "Format_Print.hpp"
#include <filesystem>
using namespace nvinfer1;
using namespace cv;
using namespace std;
using namespace sample;

/*
 *  @brief 检查断言
 *
 *  @param expr 表达式
 *
 *  @note   若表达式为真，则打印错误信息并退出程序
 */
#define Assert(expr)                                       \
    do                                                     \
    {                                                      \
        if (expr)                                          \
        {                                                  \
            printf(__CLEAR__                               \
                       __HIGHLIGHT__ __FRED__ #expr "\n"); \
            exit(-1);                                      \
        }                                                  \
    } while (0)

// #expr可以将expr替换为对应的字符串。

/*形参包展开，真tm难用*/
/*注意constexpt使得表达式在编译期确定从而使之无空递归条件*/

/*
 *  @brief 最大值和最小值求解模板
 *
 *  @param firstarg 第一个参数
 *  @param args     其他参数
 *
 *  @return 最大值或最小值
 *
 */

template <class T, class... Tp>
T reduce_max(T firstarg, Tp... args)
{
    if constexpr (sizeof...(args) <= 0)
        return firstarg;
    else
        return max(firstarg, reduce_max(args...));
}

template <class T, class... Tp>
T reduce_min(T firstarg, Tp... args)
{
    if constexpr (sizeof...(args) <= 0)
        return firstarg;
    else
        return min(firstarg, reduce_min(args...));
}

/*
 *  @brief 打印信息模板
 *
 *  @param info 信息字符串
 *
 *  @return none
 *
 */
inline void PrintInfo(const char info[])
{
    printf(__CLEAR__ __FBLUE__ __HIGHLIGHT__
           "[INFO] : %s \n" __CLEAR__,
           info);
}

/*
 *  @brief 获取维度大小
 *
 *  @param dims 维度
 *
 *  @return 维度大小
 *
 */
size_t get_dims_size(Dims dims)
{
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; i++)
    {
        size *= dims.d[i];
    }
    return size;
}

/*
 *  @brief 反sigmoid函数
 *
 *  @param 经过sigmoid函数处理的输入量
 *
 *  @return 未经过sigmoid函数输出量
 *
 */
inline constexpr float inv_sigmoid(float x)
{
    return -log(1 / x - 1);
}

/*
 *  @brief sigmoid函数
 *
 *  @param 未经过sigmoid函数输出量
 *
 *  @return 经过sigmoid函数处理的输入量
 *
 */
inline constexpr float sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}
/*
 *  @brief argmax操作
 *
 *  @param vec需要argmax的内存首
 *  @param len vec的长度
 *
 *  @return argmax的索引
 *
 */
int argmax(float *vec, int len)
{
    int max_idx = -1;
    float max_val = -0x3f3f3f3f;
    for (int i = 0; i < len; i++)
    {
        (vec[i] > max_val ? max_idx = i, max_val = vec[i] : false);
    }
    return max_idx;
}

/*
 *  @brief hwc转chw
 *
 *  @param image 需要转换的图像
 *
 *  @return none
 *
 */
void hwc2chw(Mat &image)
{
    int h = image.rows;
    int w = image.cols;
    int c = image.channels();
    image = image.reshape(1, h * w);
    image = image.t();
    image = image.reshape(w, c);
}

/*
 *  @brief 默认构造函数
 *
 */
TRTFrame::TRTFrame() : outputDims{0, 0, 0}, param(), decoder()
{

    engine = nullptr;
    device_buffer[0] = nullptr;
    device_buffer[1] = nullptr;
    host_buffer = nullptr;
    serialized_engine = nullptr;
    stream = 0;
    inputsz = 0;
    outputsz = 0;
}

/*
 *  @brief 构造函数，从onnx文件构造引擎
 *
 *  @param onnx_file  onnx文件路径
 *  @param param_     运行参数
 *
 *  @return none
 *
 */
TRTFrame::TRTFrame(const string &onnx_file, const InferParam &param_) : param(param_), decoder(MODEL_WIDTH, MODEL_HEIGHT, STRIDES, ANCHORS)
{
    filesystem::path onnx_file_path(onnx_file);
    auto engine_file_path = onnx_file_path;
    engine_file_path.replace_extension("engine");
    if (filesystem::exists(engine_file_path))
    {
        Create_Engine_From_Serialization((const string)engine_file_path.c_str());
    }
    else if (filesystem::exists(onnx_file_path))
    {
        Create_Engine_From_Onnx(onnx_file);

        Save_Serialized_Engine(engine_file_path);
    }
    else
    {
        PrintInfo("Can not find onnx file or engine file");
        Assert(true);
    }
    Assert(engine == nullptr);
    context = engine->createExecutionContext();
    Assert(context == nullptr);
    auto inputdims = engine->getBindingDimensions(engine->getBindingIndex(input_name.c_str()));
    auto outputdims = engine->getBindingDimensions(engine->getBindingIndex(output_name.c_str()));
    outputDims = Dim3d(outputdims);
    inputsz = get_dims_size(inputdims);
    outputsz = get_dims_size(outputdims);
    Assert((cudaStreamCreate(&stream)) != cudaSuccess);
    Assert(cudaMalloc(&device_buffer[0], inputsz * sizeof(float)) != cudaSuccess);
    Assert(cudaMalloc(&device_buffer[1], inputsz * sizeof(float)) != cudaSuccess);
    Assert((host_buffer = new float[outputsz]) == nullptr);
}

/*
 *  @brief 析构函数，释放资源
 *
 *  @param none
 *
 *  @return none
 *
 */
TRTFrame::~TRTFrame()
{
    if (host_buffer != nullptr)
        delete[] host_buffer;
    if (device_buffer[0] != nullptr)
        cudaFree(device_buffer[0]);
    if (device_buffer[1] != nullptr)
        cudaFree(device_buffer[1]);
    if (stream != 0)
        cudaStreamDestroy(stream);
    if (engine != nullptr)
        delete engine;
    if (serialized_engine != nullptr)
        delete serialized_engine;
}

/*
 *  @brief 从onnx文件构造引擎
 *
 *  @param onnx_file  onnx文件路径
 *
 *  @return none
 *
 */
void TRTFrame::Create_Engine_From_Onnx(const string &onnx_file)
{
    PrintInfo("Create engine from onnx file");
    auto builder = createInferBuilder(gLogger);
    Assert(builder == nullptr);
    auto network = builder->createNetworkV2(1U << static_cast<int32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    Assert(network == nullptr);
    auto parser = nvonnxparser::createParser(*network, gLogger);
    Assert(parser == nullptr);
    bool parser_success = parser->parseFromFile(onnx_file.c_str(), static_cast<int>(ILogger::Severity::kINFO));
    Assert(parser_success == false);
    if (param.topk)
    {
        /*select topk*/
        auto raw_output = network->getOutput(0);
        auto slice_layer = network->addSlice(*raw_output, Dims3{0, 0, param.conf_pos}, Dims3{1, outputDims.dim2, 1}, Dims3{1, 1, 1});
        auto raw_conf = slice_layer->getOutput(0);
        auto shuffle_layer = network->addShuffle(*raw_conf);
        shuffle_layer->setReshapeDimensions(Dims2{1, outputDims.dim2});
        raw_conf = shuffle_layer->getOutput(0);
        auto topk_layer = network->addTopK(*raw_conf, TopKOperation::kMAX, param.topk_num, 1 << 1);
        auto topk_idx = topk_layer->getOutput(1);
        auto gather_layer = network->addGather(*raw_output, *topk_idx, 1);
        gather_layer->setNbElementWiseDims(1);
        auto output_topk = gather_layer->getOutput(0);
        output_topk->setName(output_name.c_str());
        network->getInput(0)->setName(input_name.c_str());
        network->markOutput(*output_topk);
        network->unmarkOutput(*raw_output);
    }
    else
    {
        network->getInput(0)->setName(input_name.c_str());
        network->getOutput(0)->setName(output_name.c_str());
    }
    auto config = builder->createBuilderConfig();
    if (builder->platformHasFastFp16())
    {
        PrintInfo("Platform support FP16, enable FP16");
        config->setFlag(BuilderFlag::kFP16);
    }
    else
        PrintInfo("Plantform do not support FP16, enable FP32");
    size_t free, total;
    cuMemGetInfo_v2(&free, &total);
    PrintInfo(((string) "Total gpu mem : " + to_string(total >> 20) + "MB" + (string) " free gpu mem : " + to_string(free >> 20) + +"MB").c_str());
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, free);
    engine = builder->buildEngineWithConfig(*network, *config);
    /*mem free*/
    delete config;
    delete parser;
    delete network;
    delete builder;
}

/*
 *  @brief 从onnx文件构造序列化的引擎
 *
 *  @param onnx_file  onnx文件路径
 *
 *  @return none
 *
 */
void TRTFrame::Create_Serialized_Engine(const string &onnx_file)
{
    PrintInfo("Create serialized_engine from onnx file");
    auto builder = createInferBuilder(gLogger);
    Assert(builder == nullptr);
    auto network = builder->createNetworkV2(1U << static_cast<int32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    Assert(network == nullptr);
    auto parser = nvonnxparser::createParser(*network, gLogger);
    Assert(parser == nullptr);
    bool parser_success = parser->parseFromFile(onnx_file.c_str(), static_cast<int>(ILogger::Severity::kINFO));
    Assert(parser_success == false);
    if (param.topk)
    {
        /*select topk*/
        auto raw_output = network->getOutput(0);
        auto slice_layer = network->addSlice(*raw_output, Dims3{0, 0, param.conf_pos}, Dims3{1, outputDims.dim2, 1}, Dims3{1, 1, 1});
        auto raw_conf = slice_layer->getOutput(0);
        auto shuffle_layer = network->addShuffle(*raw_conf);
        shuffle_layer->setReshapeDimensions(Dims2{1, outputDims.dim2});
        raw_conf = shuffle_layer->getOutput(0);
        auto topk_layer = network->addTopK(*raw_conf, TopKOperation::kMAX, param.topk_num, 1 << 1);
        auto topk_idx = topk_layer->getOutput(1);
        auto gather_layer = network->addGather(*raw_output, *topk_idx, 1);
        gather_layer->setNbElementWiseDims(1);
        auto output_topk = gather_layer->getOutput(0);
        output_topk->setName(output_name.c_str());
        network->getInput(0)->setName(input_name.c_str());
        network->markOutput(*output_topk);
        network->unmarkOutput(*raw_output);
    }
    else
    {
        network->getInput(0)->setName(input_name.c_str());
        network->getOutput(0)->setName(output_name.c_str());
    }
    auto config = builder->createBuilderConfig();
    if (builder->platformHasFastFp16())
    {
        PrintInfo("Platform support FP16, enable FP16");
        config->setFlag(BuilderFlag::kFP16);
    }
    else
        PrintInfo("Plantform do not support FP16, enable FP32");
    size_t free, total;
    cuMemGetInfo_v2(&free, &total);
    PrintInfo(((string) "Total gpu mem : " + to_string(total >> 20) + (string) "free gpu mem : " + to_string(free >> 20)).c_str());
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, free);
    serialized_engine = builder->buildSerializedNetwork(*network, *config);
    engine = createInferRuntime(gLogger)->deserializeCudaEngine(serialized_engine->data(), serialized_engine->size());
    /*mem free*/
    delete config;
    delete parser;
    delete network;
    delete builder;
}

/*
 *  @brief 从序列化的引擎文件构造引擎
 *
 *  @param onnx_file  序列化的引擎文件路径
 *
 *  @return none
 *
 */
void TRTFrame::Create_Engine_From_Serialization(const string &onnx_file)
{
    PrintInfo("Create engine from serialized_engine file");
    std::ifstream fs(onnx_file, ios::binary);
    fs.seekg(0, ios::end);
    size_t sz = fs.tellg();
    fs.seekg(0, ios::beg);
    char *buffer = new char[sz];
    fs.read(buffer, sz);
    auto runtime = createInferRuntime(gLogger);
    Assert(runtime == nullptr);
    Assert((engine = runtime->deserializeCudaEngine(buffer, sz)) == nullptr);
    delete[] buffer;
    runtime->destroy();
}

/*
 *  @brief 保存序列化的引擎文件
 *
 *  @param des  保存路径
 *
 *  @return none
 *
 */
void TRTFrame::Save_Serialized_Engine(const string &des)
{
    auto buffer_serialized_engine = engine->serialize();
    Assert(buffer_serialized_engine == nullptr);
    ofstream fs(des, ios::binary);
    fs.write(static_cast<const char *>(buffer_serialized_engine->data()), buffer_serialized_engine->size());
    delete buffer_serialized_engine;
}

/*
 *  @brief 保存序列化的引擎文件
 *
 *  @param serialized_engine_  序列化的引擎
 *  @param des  保存路径
 *
 *  @return none
 *
 */
void TRTFrame::Save_Serialized_Engine(IHostMemory *serialized_engine_, const string &des)
{
    Assert(serialized_engine_ == nullptr);
    ofstream fs(des, ios::binary);
    fs.write(static_cast<const char *>(serialized_engine_->data()), serialized_engine_->size());
    delete serialized_engine_;
}

/*
 *  @brief 推理,并将结果存入private成员host_buffer中
 *
 *  @param input_tensor  输入数据
 *
 *  @return none
 *
 */
void TRTFrame::Infer(void *input_tensor)
{
    cudaMemcpyAsync(device_buffer[0], input_tensor, inputsz * sizeof(float), cudaMemcpyHostToDevice, stream);
    context->setOptimizationProfileAsync(0, stream);
    context->setTensorAddress(input_name.c_str(), device_buffer[0]);
    context->setTensorAddress(output_name.c_str(), device_buffer[1]);
    Assert(context->enqueueV3(stream) == false);
    cudaMemcpyAsync(host_buffer, device_buffer[1], outputsz * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}

/*
 *  @brief 计算IOU，格式为xyxyxyxy
 *
 *  @param xyxyxyxy1  第一个box的四点坐标
 *  @param xyxyxyxy2  第二个box的四点坐标
 *
 *  @return IOU值
 */
float TRTFrame::IOU_xyxyxyxy(float xyxyxyxy1[8], float xyxyxyxy2[8])
{
    Rect2f box1, box2;
    box1.x = reduce_min(xyxyxyxy1[0], xyxyxyxy1[2], xyxyxyxy1[4], xyxyxyxy1[6]);
    box1.y = reduce_min(xyxyxyxy1[1], xyxyxyxy1[3], xyxyxyxy1[5], xyxyxyxy1[7]);
    box1.width = reduce_max(xyxyxyxy1[0], xyxyxyxy1[2], xyxyxyxy1[4], xyxyxyxy1[6]) - box1.x;
    box1.height = reduce_max(xyxyxyxy1[1], xyxyxyxy1[3], xyxyxyxy1[5], xyxyxyxy1[7]) - box1.y;

    box2.x = reduce_min(xyxyxyxy2[0], xyxyxyxy2[2], xyxyxyxy2[4], xyxyxyxy2[6]);
    box2.y = reduce_min(xyxyxyxy2[1], xyxyxyxy2[3], xyxyxyxy2[5], xyxyxyxy2[7]);
    box2.width = reduce_max(xyxyxyxy2[0], xyxyxyxy2[2], xyxyxyxy2[4], xyxyxyxy2[6]) - box2.x;
    box2.height = reduce_max(xyxyxyxy2[1], xyxyxyxy2[3], xyxyxyxy2[5], xyxyxyxy2[7]) - box2.y;

    float Intersection = (box1 & box2).area();
    float Union = box1.area() + box2.area() - Intersection;
    return Intersection / Union;
}

/*
 *  @brief 计算IOU，格式为xywh_topl
 *
 *  @param xywh1  第一个box的左上角(top,left)坐标和宽高
 *  @param xywh2  第二个box的左上角(top,left)坐标和宽高
 *
 *  @return IOU值
 *
 */

float TRTFrame::IOU_xywh_topl(float xywh1[4], float xywh2[4])
{
    Rect2f box1(Point2f(xywh1[1], xywh1[0]), Size(xywh1[2], xywh1[3]));
    Rect2f box2(Point2f(xywh2[1], xywh2[0]), Size(xywh2[2], xywh2[3]));
    float Intersection = (box1 & box2).area();
    float Union = box1.area() + box2.area() - Intersection;
    return Intersection / Union;
}

/*
 *  @brief 计算IOU，格式为xywh_center
 *
 *  @param xywh1  第一个box的中心坐标和宽高
 *  @param xywh2  第二个box的中心坐标和宽高
 *
 *  @return IOU值
 *
 */
float TRTFrame::IOU_xywh_center(float xywh1[4], float xywh2[4])
{
    xywh1[0] = xywh1[0] - xywh1[2] / 2;
    xywh1[1] = xywh1[1] - xywh1[3] / 2;
    xywh2[0] = xywh2[0] - xywh2[2] / 2;
    xywh2[1] = xywh2[1] - xywh2[3] / 2;
    return IOU_xywh_topl(xywh1, xywh2);
}

/*
 *  @brief 计算IOU，格式为xyxy
 *
 *  @param xyxy1  第一个box的左上角和右下角坐标
 *  @param xyxy2  第二个box的左上角和右下角坐标
 *
 *  @return IOU值
 *
 */
float TRTFrame::IOU_xyxy(float xyxy1[4], float xyxy2[4])
{
    Rect2f box1(Point2f(xyxy1[0], xyxy1[1]), Point2f(xyxy1[2], xyxy1[3]));
    Rect2f box2(Point2f(xyxy2[0], xyxy2[1]), Point2f(xyxy2[2], xyxy2[3]));

    float Intersection = (box1 & box2).area();
    float Union = box1.area() + box2.area() - Intersection;
    return Intersection / Union;
}

/*
 *  @brief 计算IOU
 *
 *  @param pts1  第一个box的坐标
 *  @param pts2  第二个box的坐标
 *  @param type  坐标格式
 *
 *  @return IOU值
 *
 */
float TRTFrame::IOU(float *pts1, float *pts2, box_type type)
{
    switch (type)
    {
    case xyxyxyxy:
        return IOU_xyxyxyxy(pts1, pts2);
    case xyhw_center:
    {
        float value = IOU_xywh_center(pts1, pts2);
        pts1[0] = pts1[0] + pts1[2] / 2;
        pts1[1] = pts1[1] + pts1[3] / 2;
        pts2[0] = pts2[0] + pts2[2] / 2;
        pts2[1] = pts2[1] + pts2[3] / 2;
        return value;
    }
    case xyhw_topl:
        return IOU_xywh_topl(pts1, pts2);
    case xyxy:
        return IOU_xyxy(pts1, pts2);
    }
    return 0.0f;
}

/*
 *  @brief 非极大值抑制，输入采用外部数据
 *
 *  @param output_tensor  输出张量
 *  @param res_tensor  输出结果
 *
 *  @return none
 *
 */
void TRTFrame::NMS(vector<float> &output_tensor, vector<vector<float>> &res_tensor)
{
    float conf_thre = param.conf_thre;
    if (!param.has_sigmoid)
        conf_thre = inv_sigmoid(param.conf_thre);
    res_tensor.clear();
    vector<vector<float>> tmp_store;
    for (int i = 0; i < outputDims.dim2; i++)
    {
        if (output_tensor[i * outputDims.dim3 + param.conf_pos] < param.conf_thre)
            if (!param.topk)
                continue;
            else
                break;
        tmp_store.emplace_back(
            vector<float>(output_tensor.begin() + i * outputDims.dim3,
                          output_tensor.begin() + i * outputDims.dim3 + outputDims.dim3));
    }
    if (!param.topk)
        sort(tmp_store.begin(), tmp_store.end(),
             [this](vector<float> box1, vector<float> box2)
             { return box1[param.conf_pos] > box2[param.conf_pos]; });
    vector<float> Res;
    vector<bool> Removed(outputDims.dim2, false);
    for (int i = 0; i < tmp_store.size(); i++)
    {
        if (!Removed[i])
        {
            Res = tmp_store[i];
            Removed[i] = true;
        }
        else
            continue;
        for (int j = i + 1; j < tmp_store.size(); j++)
        {
            if (!Removed[j])
            {
                float iou = IOU(&Res[0] + param.box_pos, &tmp_store[j][0] + param.box_pos, param.type);
                if (iou > param.iou_thre)
                    Removed[j] = true;
            }
        }
        res_tensor.emplace_back(Res);
    }
}

/*
 *  @brief 非极大值抑制，输入采用私有变量host_buffer
 *
 *  @param res_tensor  NMS后的各个tensor的信息
 *
 *  @return none
 */
void TRTFrame::NMS(vector<vector<float>> &res_tensor)
{
    Assert(host_buffer == nullptr);
    if (param.isAnchor)
        decoder.decodeOutputs(host_buffer, outputDims.dim2, outputDims.dim3, param.box_pos);
    float conf_thre = param.conf_thre;
    // if (!param.has_sigmoid)
    //     conf_thre = inv_sigmoid(param.conf_thre);
    res_tensor.clear();
    for (int i = 0; i < outputDims.dim2; i++)
    {
        if (host_buffer[i * outputDims.dim3 + param.conf_pos] < conf_thre)
            if (!param.topk)
                continue;
            else
                break;
        else
            res_tensor.emplace_back(host_buffer + i * outputDims.dim3, host_buffer + i * outputDims.dim3 + outputDims.dim3);
    }

    if (!param.topk)
        sort(res_tensor.begin(), res_tensor.end(),
             [this](vector<float> box1, vector<float> box2)
             { return box1[param.conf_pos] > box2[param.conf_pos]; });

    vector<bool> removed(res_tensor.size(), false);

    for (int i = 0; i < res_tensor.size(); i++)
    {
        if (removed[i])
            continue;
        for (int j = i + 1; j < res_tensor.size(); j++)
        {
            if (IOU(&res_tensor[i][param.box_pos], &res_tensor[j][param.box_pos], param.type) > param.iou_thre)
                removed[j] = true;
        }
    }

    int back_idx = 0;
    for (int i = 0; i < res_tensor.size(); i++)
        if (removed[i])
            swap(res_tensor[i], *(&res_tensor.back() - back_idx++));
    res_tensor.erase(res_tensor.end() - back_idx, res_tensor.end());
}

/*
 *  @brief 预处理，将输入图像转换为输入格式
 *
 *  @param src  输入图像
 *  @param blob  输出格式
 *
 *  @return none
 *
 */

void TRTFrame::Preprocess(Mat &src, Mat &blob)
{
    fx = src.cols / (float)param.input_size.width;
    fy = src.rows / (float)param.input_size.height;
    resize(src, blob, param.input_size);
    cvtColor(blob, blob, param.cvt_code);
    blob.convertTo(blob, CV_32F);
    if (param.normalize)
        blob /= 255.0;
    if (param.hwc2chw)
        hwc2chw(blob);
}

/*
 *  @brief 后处理，将输出结果转换为BoxInfo格式
 *
 *  @param res_tensor  NMS后的各个tensor的信息
 *  @param box_infos  输出结果
 *
 *  @return none
 *
 */
void TRTFrame::Postprocess(std::vector<std::vector<float>> &res_tensor, vector<BoxInfo> &box_infos)
{
    box_infos.clear();

    for (auto &vec : res_tensor)
    {
        for (int i = param.box_pos; i < (param.type == xyxyxyxy ? 8 : 4); i++)
            (i % 2 == 0 ? vec[i] *= fx : vec[i] *= fy);
        BoxInfo info(vec[param.conf_pos], &vec[param.box_pos], param.type);
        for (auto &classes : param.classes_info)
        {
            int class_idx = argmax(&vec[classes.classes_offset], classes.classes_num);
            info.classes.emplace_back(pair<int, string>(class_idx, classes.classes_names[class_idx]));
        }
        box_infos.emplace_back(info);
    }
}

/*
 *  @brief 显示结果，格式为xyxyxyxy
 *
 *  @param box_infos 后处理后的结果
 *  @param img  输入图像
 *
 *  @return none
 */
void TRTFrame::Show_xyxyxyxy(Mat &img, vector<BoxInfo> &box_infos)
{
    for (const auto &info : box_infos)
    {
        line(img, Point(info.box[0], info.box[1]), Point(info.box[2], info.box[3]), Scalar(0, 255, 0), 2);
        line(img, Point(info.box[2], info.box[3]), Point(info.box[4], info.box[5]), Scalar(0, 255, 0), 2);
        line(img, Point(info.box[4], info.box[5]), Point(info.box[6], info.box[7]), Scalar(0, 255, 0), 2);
        line(img, Point(info.box[6], info.box[7]), Point(info.box[0], info.box[1]), Scalar(0, 255, 0), 2);
    }
}

/*
 *  @brief 显示结果，格式为xywh_center
 *
 *  @param img  输入图像
 *  @param box_infos 后处理后的结果
 *
 *  @return none
 */

void TRTFrame::Show_xywh_center(Mat &img, vector<BoxInfo> &box_infos)
{
    for (const auto &info : box_infos)
    {
        float x1 = info.box[0] - info.box[2] / 2;
        float y1 = info.box[1] - info.box[3] / 2;
        float x2 = info.box[0] + info.box[2] / 2;
        float y2 = info.box[1] + info.box[3] / 2;
        rectangle(img, Point(x1, y1), Point(x2, y2), myColor[info.classes[0].first], 2);
        putText(img, info.classes[0].second, Point(x1, y1), FONT_HERSHEY_SIMPLEX, 1, myColor[info.classes[0].first], 2);
    }
}

void TRTFrame::Show(Mat &img, vector<BoxInfo> &box_infos, box_type type)
{
    switch (type)
    {
    case xyhw_center:
        Show_xywh_center(img, box_infos);
        break;
    case xyxyxyxy:
        Show_xyxyxyxy(img, box_infos);
        break;
    }
}
/*
 *  @brief  启动函数
 *
 *  @param src：输入图像
 *  @param box_infos: 输出各个框的信息
 *
 *  @return none
 *
 */
void TRTFrame::Run(Mat &src, vector<BoxInfo> &box_infos)
{
    Mat blob;
    box_infos.clear();

    /*预处理*/
    Preprocess(src, blob);
    /*推理*/
    Infer(blob.data);

    /*后处理*/
    vector<vector<float>> res_tensor;
    NMS(res_tensor);
    Postprocess(res_tensor, box_infos);

    /*显示结果*/
    Show(src, box_infos, param.type);
}

void yolov5OutputDecoder::generateGrid(int width, int height, vector<yolov5OutputDecoder::pairFloat> &grids)
{
    grids.clear();
    grids.resize(width * height);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int idx = y * width + x;
            grids[idx].first = static_cast<float>(x);
            grids[idx].second = static_cast<float>(y);
        }
    }
}

/*  写在前面，这是一个用于从yolov5的xywh格式进行解码的函数
 *
 *  记录一下为何需要这个函数
 *
 *  对于yolo的输出数据的维度一般如下
 *
 *  1 * nbOutputs * nbProperty
 *
 *  其中nbOutpus即为输出的框的个数，而nbProperty即为每个框所对应的属性
 *
 *  nbProperty的排列如下 [x, y, w, h, conf, class1, class2, ...]
 *
 *  x, y, w, h即为预测框的中心坐标以及宽高， conf为置信度， class1, class2, ...为类别置信度
 *
 *  也就是说conf是确保这个东西是一个物体的概率，而class1, class2, ...是这个物体属于某一类的概率
 *
 *  对于xywh，输出的并非直接是图片上的单位像素，而是预测的相对于grid的偏移量。
 *
 *  所谓的grid，就是将图片划分为一个个的小格子，每个格子对应一个预测框，而这个预测框的坐标就是相对于这个格子的偏移量
 *
 *  yolov5将一张图片通过步长(stride)分为不同大小的特征图，以确保不同大小的物体都能被检测到
 *
 *  步长固定为32，16，8，分别对应三个不同大小的特征图，也就是说，以640*640的图片为例，分别对应20*20(640/32), 40*40(640/16), 80*80(640/8)的特征图
 *
 *  如何理解呢？
 *
 *  以stride(别忘了是步长)为16，也就是说每16个像素进行一次下采样
 *
 *  所谓的下采样就是通过某种规定的方法来综合一定范围内的像素，从而得到一个新的像素。其中有最大池化(取范围内最大值最为范围内像素的特征)等，
 *  显然yolov5的下采样的方法会增加综合和复杂，但是这里不是重点
 *
 *  对于16的步长，每次对应原图中16*16的区域，下采样后这片区域用一个像素作为其特征。
 *
 *  对于640*640的图片，实现步长16的下采样后，单看一列，那么就应该有640 / 16 = 40个像素特征值，因此得到的特征图整个的尺寸为40*40
 *
 *  同理，对于步长32，得到的特征图尺寸为20*20，对于步长8，得到的特征图尺寸为80*80
 *
 *  而yolo通过这些特征图来进行预测，也就是说，对于每个特征图上的每个像素，都会有一个预测框，这个预测框的坐标就是相对于这个像素的偏移量
 *
 *  或者说是相对于这个grid cell的偏移量,每个gridcell就对应原图中stride*stride的区域
 *
 *  而由于预测的物体的不同，对于每一维度的特征图，都有着不同的anchor box，也就是说，对于每一个grid cell，都会有多个anchor box的尺寸
 *  对于yolov5,其尺寸如下
 *   [(10, 13), (16, 30), (33, 23)],  # P3/8，stride为8的anchor尺寸为长宽分别为(10,13), (16,30), (33,23)的三种矩形
 *   [(30, 61), (62, 45), (59, 119)],  # P4/16
 *   [(116, 90), (156, 198), (373, 326)]  # P5/32
 *   也就是说，对于每一stride的gridcell，都会有三种anchor box的尺寸
 *
 *  因此我们预测出的wh参数也并非相对于原图640*640,而是相对于每一anchorbox的尺寸的偏移量
 *
 *  在生成最终结果时，可以理解为如此
 *
 *  原图 640*640, num = 1 -> 特征图，下采样8, gridcell个数=640/8=80 * 80, 下采样16, gridcell个数=640/16=40, 下采样32, gridcell个数=640/32=20
 *  num即为grid cell的数量，也就是说一共有80*80 + 40*40 + 20*20个grid cell，一共8400个grid cell
 *  而每个grid cell都会有三种anchor box的尺寸，因此一共有8400*3=25200个预测框，这些框的排列顺序也是按照8, 16, 32的下采样顺序排列的
 *
 *  对于不同的输入尺寸，其grid cell的数量也会不同，但是其anchor box的数量是固定的。对于320*320的输入，就变为了
 *  40*40 + 20*20 + 10*10 = 2100个grid cell，一共6300个预测框
 *
 *  根据yolo官方给出的公式，我们便可将预测值的偏移量转换到实际尺寸
 *  ps:x,y,w,h的预测值均应经过sigmoid函数处理
 *  x = (x * 2 - 0.5 + grid_x) * stride
 *  y = (y * 2 - 0.5 + grid_y) * stride
 *  w = (w * 2) ^ 2 * anchor_w
 *  h = (h * 2) ^ 2 * anchor_h
 */

void yolov5OutputDecoder::decodeOutputs(float *outputs, int allLen, int dataLen, int boxPos)
{
    int rowIdx = 0;
    for (int layer = 0; layer < m_nbLayers; layer++)
    {
        int gridWidth = m_modelWidth / m_strides[layer];
        int gridHeight = m_modelHeight / m_strides[layer];
        int gridSz = gridWidth * gridHeight;
        int nbPredictions = gridSz * m_nbAnchors;

        if (m_grids[layer].empty() || m_grids[layer].size() != gridSz)
            generateGrid(gridWidth, gridHeight, m_grids[layer]);
#pragma omp parallel for if (nbPredictions > 1000)
        for (int i = 0; i < nbPredictions; i++)
        {
            int currentRow = rowIdx + i;
            int currentGrid = i % gridSz;
            int currentAnchor = i / gridSz;

            outputs[currentRow * dataLen + boxPos] = (outputs[currentRow * dataLen + boxPos] * 2 - 0.5 + m_grids[layer][currentGrid].first) * m_strides[layer];
            outputs[currentRow * dataLen + boxPos + 1] = (outputs[currentRow * dataLen + boxPos + 1] * 2 - 0.5 + m_grids[layer][currentGrid].second) * m_strides[layer];

            outputs[currentRow * dataLen + boxPos + 2] = pow(outputs[currentRow * dataLen + boxPos + 2] * 2, 2) * m_anchorsGrids[layer][currentAnchor].first;
            outputs[currentRow * dataLen + boxPos + 3] = pow(outputs[currentRow * dataLen + boxPos + 3] * 2, 2) * m_anchorsGrids[layer][currentAnchor].second;
        }
        rowIdx += nbPredictions;
    }
}

void yolov5OutputDecoder::decodeOutputs(vector<float> &outputs, int dataLen, int boxPos)
{
    decodeOutputs(outputs.data(), outputs.size(), dataLen, boxPos);
}

void yolov5OutputDecoder::decodeOutputs(vector<vector<float>> &outputs, int boxPos)
{
    int rowIdx = 0;

    for (int layer = 0; layer < m_nbLayers; layer++)
    {
        int gridWidth = m_modelWidth / m_strides[layer];   // 特征图宽
        int gridHeight = m_modelHeight / m_strides[layer]; // 特征图高
        int gridSz = gridWidth * gridHeight;               // 特征图大小或者说原图该步长的grid cell数量
        int nbPredictions = gridSz * m_nbAnchors;          // 每个特征图上的预测框数量

        if (m_grids[layer].empty() || m_grids[layer].size() != gridSz)
            generateGrid(gridWidth, gridHeight, m_grids[layer]);
#pragma omp parallel for if (nbPredictions > 1000)
        for (int i = 0; i < nbPredictions; i++)
        {
            int currentRow = rowIdx + i;
            int currentGrid = i % gridSz;   // 横向增长，每次增长一个grid cell，这相当于当前grid cell的索引，比如 1->(0,1),2->(0,2)
            int currentAnchor = i / gridSz; // 当前gridcell对应的anchor尺寸
            // 排列为 gridSz 个第一个anchor， gridSz 个第二个anchor， gridSz 个第三个anchor

            outputs[currentRow][boxPos] = (outputs[currentRow][boxPos] * 2 - 0.5 + m_grids[layer][currentGrid].first) * m_strides[layer];
            outputs[currentRow][boxPos + 1] = (outputs[currentRow][boxPos + 1] * 2 - 0.5 + m_grids[layer][currentGrid].second) * m_strides[layer];

            outputs[currentRow][boxPos + 2] = pow(outputs[currentRow][boxPos + 2] * 2, 2) * m_anchorsGrids[layer][currentAnchor].first;
            outputs[currentRow][boxPos + 3] = pow(outputs[currentRow][boxPos + 3] * 2, 2) * m_anchorsGrids[layer][currentAnchor].second;
        }
        rowIdx += nbPredictions;
    }
}