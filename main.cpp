#include "TRTFrame.hpp"
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <ctime>
using namespace std;
using namespace cv;

const string onnx_file = "../model-opt-4.onnx";
int main()
{
    // pre-process [bgr2rgb & resize]
    Mat src;
    TRTFrame trt(onnx_file);
    cout << "pass" << endl;
    VideoCapture cap("../test.mp4");

    while (true)
    {
        cap.read(src);
        cv::Mat x;
        float fx = (float)src.cols / 640.f, fy = (float)src.rows / 384.f; // 求出resize前的比例，用于之后将四点从标准输入上的尺寸转换到输入上的尺寸

        /*图像进行转换成RGB，然后Resize*/
        cv::cvtColor(src, x, cv::COLOR_BGR2RGB);

        if (src.cols != 640 || src.rows != 384)
        {
            cv::resize(x, x, {640, 384});
        }
        Mat x_;
        resize(src, x_, {640, 384});
        x.convertTo(x, CV_32F);

        auto out = trt.Infer(x.data);

        vector<vector<float>> res;
        trt.NMS(xyxyxyxy, out, res, nmspara{.conf_pos = 8, .box_pos = 0, .conf_thre = 0.1, .iou_thre = 0.4, .has_sigmoid = false});
        for (auto &r : res)
        {
            line(x_, Point2f(r[0], r[1]), Point2f(r[2], r[3]), Scalar(0, 255, 0), 2);
            line(x_, Point2f(r[2], r[3]), Point2f(r[4], r[5]), Scalar(0, 255, 0), 2);
            line(x_, Point2f(r[4], r[5]), Point2f(r[6], r[7]), Scalar(0, 255, 0), 2);
            line(x_, Point2f(r[6], r[7]), Point2f(r[0], r[1]), Scalar(0, 255, 0), 2);
        }
        cv::imshow("src", x_);
        cv::waitKey(10);
    }
    return 0;
}