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

/*形参包展开，真tm难用*/
/*注意constexpt使得表达式在编译期确定从而使之无空递归条件*/
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

inline void PrintInfo(const char info[])
{
    printf(__CLEAR__ __FBLUE__ __HIGHLIGHT__
           "[INFO] : %s \n" __CLEAR__,
           info);
}

size_t get_dims_size(Dims dims)
{
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; i++)
    {
        size *= dims.d[i];
    }
    return size;
}

inline constexpr float inv_sigmoid(float x)
{
    return -log(1 / x - 1);
}

TRTFrame::TRTFrame() : outputDims{0, 0, 0}
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

TRTFrame::TRTFrame(const string &onnx_file)
{
    filesystem::path onnx_file_path(onnx_file);
    auto engine_file_path = onnx_file_path;
    engine_file_path.replace_extension("engine");
    if (filesystem::exists(engine_file_path))
    {
        auto st = Create_Engine_From_Serialization((const string)engine_file_path.c_str());
    }
    else
    {
        Create_Engine_From_Onnx(onnx_file);

        Save_Serialized_Engine(engine_file_path);
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

bool TRTFrame::Create_Engine_From_Onnx(const string &onnx_file)
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
    network->getInput(0)->setName(input_name.c_str());
    network->getOutput(0)->setName(output_name.c_str());
    auto config = builder->createBuilderConfig();
    if (builder->platformHasFastFp16())
        PrintInfo("Platform support FP16, enable FP16");
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
    return true;
}

bool TRTFrame::Create_Serialized_Engine(const string &onnx_file)
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
    network->getInput(0)->setName(input_name.c_str());
    network->getOutput(0)->setName(output_name.c_str());
    auto config = builder->createBuilderConfig();
    if (builder->platformHasFastFp16())
        PrintInfo("Platform support FP16, enable FP16");
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
    return true;
}

bool TRTFrame::Create_Engine_From_Serialization(const string &onnx_file)
{
    PrintInfo("Create engine from serialized_engine file");
    std::ifstream fs(onnx_file, ios::binary);
    fs.seekg(0, ios::end);
    size_t sz = fs.tellg();
    fs.seekg(0, ios::beg);
    auto buffer = make_unique<char[]>(sz);
    fs.read(buffer.get(), sz);
    auto runtime = createInferRuntime(gLogger);
    Assert(runtime == nullptr);
    Assert((engine = runtime->deserializeCudaEngine(buffer.get(), sz)) == nullptr);
    buffer.release();
    runtime->destroy();
    return true;
}

bool TRTFrame::Save_Serialized_Engine(const string &des)
{
    auto buffer_serialized_engine = engine->serialize();
    Assert(buffer_serialized_engine == nullptr);
    ofstream fs(des, ios::binary);
    fs.write(static_cast<const char *>(buffer_serialized_engine->data()), buffer_serialized_engine->size());
    delete buffer_serialized_engine;
    return true;
}

bool TRTFrame::Save_Serialized_Engine(IHostMemory *serialized_engine_, const string &des)
{
    Assert(serialized_engine_ == nullptr);
    ofstream fs(des, ios::binary);
    fs.write(static_cast<const char *>(serialized_engine_->data()), serialized_engine_->size());
    delete serialized_engine_;
    return true;
}

vector<float> TRTFrame::Infer(void *input_tensor)
{
    cudaMemcpyAsync(device_buffer[0], input_tensor, inputsz * sizeof(float), cudaMemcpyHostToDevice, stream);
    context->setOptimizationProfileAsync(0, stream);
    context->setTensorAddress(input_name.c_str(), device_buffer[0]);
    context->setTensorAddress(output_name.c_str(), device_buffer[1]);
    Assert(context->enqueueV3(stream) == false);
    cudaMemcpyAsync(host_buffer, device_buffer[1], outputsz * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return vector<float>(host_buffer, host_buffer + outputsz);
}

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

float TRTFrame::IOU_xywh_topl(float xyhw1[4], float xyhw2[4])
{
    Rect2f box1(Point2f(xyhw1[0], xyhw1[1]), Size(xyhw1[3], xyhw1[2]));
    Rect2f box2(Point2f(xyhw2[0], xyhw2[1]), Size(xyhw2[3], xyhw2[2]));
    float Intersection = (box1 & box2).area();
    float Union = box1.area() + box2.area() - Intersection;
    return Intersection / Union;
}

float TRTFrame::IOU_xywh_center(float xyhw1[4], float xyhw2[4])
{
    xyhw1[0] = xyhw1[0] - xyhw1[3] / 2;
    xyhw1[1] = xyhw1[1] - xyhw1[2] / 2;
    xyhw2[0] = xyhw2[0] - xyhw2[3] / 2;
    xyhw2[1] = xyhw2[1] - xyhw2[2] / 2;
    return IOU_xywh_topl(xyhw1, xyhw2);
}

float TRTFrame::IOU_xyxy(float xyxy1[4], float xyxy2[4])
{
    Rect2f box1(Point2f(xyxy1[0], xyxy1[1]), Point2f(xyxy1[2], xyxy1[3]));
    Rect2f box2(Point2f(xyxy2[0], xyxy2[1]), Point2f(xyxy2[2], xyxy2[3]));

    float Intersection = (box1 & box2).area();
    float Union = box1.area() + box2.area() - Intersection;
    return Intersection / Union;
}

float TRTFrame::IOU(float *pts1, float *pts2, box_type type)
{
    switch (type)
    {
    case xyxyxyxy:
        return IOU_xyxyxyxy(pts1, pts2);
    case xyhw_center:
        return IOU_xywh_center(pts1, pts2);
    case xyhw_topl:
        return IOU_xywh_topl(pts1, pts2);
    case xyxy:
        return IOU_xyxy(pts1, pts2);
    }
    return 0.0f;
}

void TRTFrame::NMS(box_type type, vector<float> &output_tensor, vector<vector<float>> &res_tensor, const nmspara &para)
{
    static_confpos = para.conf_pos;
    float conf_thre;
    if (!para.has_sigmoid)
        conf_thre = inv_sigmoid(para.conf_thre);
    int box_buffer_len = (type == xyxyxyxy ? 8 : 4);
    res_tensor.clear();
    vector<vector<float>> tmp_store;
    for (int i = 0; i < outputDims.dim2; i++)
    {
        if (output_tensor[i * outputDims.dim3 + static_confpos] < conf_thre)
            continue;
        tmp_store.emplace_back(
            vector<float>(output_tensor.begin() + i * outputDims.dim3,
                          output_tensor.begin() + i * outputDims.dim3 + outputDims.dim3));
    }
    sort(tmp_store.begin(), tmp_store.end(),
         [](vector<float> box1, vector<float> box2)
         { return box1[static_confpos] > box2[static_confpos]; });
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
                float iou = IOU(&Res[0] + para.box_pos, &tmp_store[j][0] + para.box_pos, type);
                if (iou > para.iou_thre)
                    Removed[j] = true;
            }
        }
        res_tensor.emplace_back(Res);
    }
}