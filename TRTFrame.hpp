#ifndef _TRTFRAME_HPP_
#define _TRTFRAME_HPP_
#include <opencv4/opencv2/core.hpp>
#include <vector>
#include "NvInfer.h"

/*
 *  @brief 定义NMS参数
 *
 *  @note conf_pos: 置信度的位置，0表示第一个元素，1表示第二个元素，以此类推
 *        box_pos: 边界框的位置，0表示第一个元素，1表示第二个元素，以此类推
 *        conf_thre: 置信度阈值，低于该阈值的边界框会被过滤掉
 *        iou_thre: IOU阈值，低于该阈值的边界框会被过滤掉
 *        has_sigmoid: 是否使用sigmoid激活函数,这取决与模型最后一层全链接是否经过sigomid激活，如果经过sigmoid激活，则置为1，否则为0
 */
struct nmspara
{
    int conf_pos;
    int box_pos;
    float conf_thre;
    float iou_thre;
    bool has_sigmoid;
};
static int static_confpos = -1;

/*
 *  @brief 定义box的类型
 *
 *  @note   xyxyxyxy: [x1, y1, x2, y2, x3, y3, x4, y4] 逆时针四点
 *          xyhw_center: [x, y, w, h, center_x, center_y]  中心点坐标和宽高
 *          xyhw_topl: [x, y, w, h, top_left_x, top_left_y]    左上角坐标和宽高
 *          xyxy: [x1, y1, x2, y2] 左上角坐标和右下角坐标
 *
 */
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

/*
 *  @brief 基于TensorRT的onnx推理框架类
 *
 *  @note  该类主要用于对TensorRT推理框架的封装，包括创建推理引擎、序列化、保存、推理、NMS等功能
 */
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
    void Create_Engine_From_Onnx(const std::string &onnx_file);
    void Create_Engine_From_Serialization(const std::string &onnx_file);
    void Create_Serialized_Engine(const std::string &onnx_file);
    void Save_Serialized_Engine(const std::string &des);
    void Save_Serialized_Engine(nvinfer1::IHostMemory *serialized_engine_, const std::string &des);
    std::vector<float> Infer(void *input_tensor);
    float IOU_xyxyxyxy(float yxyxyxy1[8], float xyxyxyxy2[8]);
    float IOU_xywh_center(float xyhw1[4], float xyhw2[4]);
    float IOU_xywh_topl(float xyhw1[4], float xyhw2[4]);
    float IOU_xyxy(float xyxy1[4], float xyxy2[4]);
    float IOU(float *pts1, float *pts2, box_type type);
    void NMS(box_type type, std::vector<float> &output_tensor, std::vector<std::vector<float>> &res_tensor, const nmspara &para);
    // preprocess;postprocess;
};

#endif