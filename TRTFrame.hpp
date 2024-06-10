#ifndef _TRTFRAME_HPP_
#define _TRTFRAME_HPP_
#include <opencv4/opencv2/core.hpp>
#include <vector>
#include "NvInfer.h"

struct nmspara
{
    int conf_pos;
    int box_pos;
    float conf_thre;
    float iou_thre;
    bool has_sigmoid;
};
static int static_confpos = -1;
enum box_type
{
    xyxyxyxy = 1,
    xyhw_center,
    xyhw_topl,
    xyxy
};

struct Dim3d
{
    int dim1;
    int dim2;
    int dim3;
    Dim3d(int dim1_, int dim2_, int dim3_)
    {
        dim1 = dim1_;
        dim2 = dim2_;
        dim3 = dim3_;
    };
    Dim3d(nvinfer1::Dims &dim)
    {
        dim1 = dim.d[0];
        dim2 = dim.d[1];
        dim3 = dim.d[2];
    }
    Dim3d()
    {
        dim1 = -1;
        dim2 = -1;
        dim3 = -1;
    }
};
class TRTFrame
{
public:
    Dim3d outputDims;
    const std::string input_name = "input";
    const std::string output_name = "output";
    size_t inputsz;
    size_t outputsz;
    nvinfer1::IHostMemory *serialized_engine;
    nvinfer1::ICudaEngine *engine;
    nvinfer1::IExecutionContext *context;
    cudaStream_t stream;
    mutable void *device_buffer[2];
    float *host_buffer;

public:
    TRTFrame();
    explicit TRTFrame(const std::string &onnx_file);
    ~TRTFrame();
    bool Create_Engine_From_Onnx(const std::string &onnx_file);
    bool Create_Engine_From_Serialization(const std::string &onnx_file);
    bool Create_Serialized_Engine(const std::string &onnx_file);
    bool Save_Serialized_Engine(const std::string &des);
    bool Save_Serialized_Engine(nvinfer1::IHostMemory *serialized_engine_, const std::string &des);
    std::vector<float> Infer(void *input_tensor);
    float IOU_xyxyxyxy(float yxyxyxy1[8], float xyxyxyxy2[8]);
    float IOU_xywh_center(float xyhw1[4], float xyhw2[4]);
    float IOU_xywh_topl(float xyhw1[4], float xyhw2[4]);
    float IOU_xyxy(float xyxy1[4], float xyxy2[4]);
    float IOU(float *pts1, float *pts2, box_type type);
    void NMS(box_type type, std::vector<float> &output_tensor, std::vector<std::vector<float>> &res_tensor, const nmspara &para);
    // preprocess;postprocess;
    // postprocess;
};

#endif