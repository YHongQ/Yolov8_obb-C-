#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>
#include<io.h>
#include<string>
#include<sstream>
#include<iomanip>
#include<time.h>

// 用于存储模型输出结果，YoLov8-OBB ,input: 1*640*640*3
// output : 1*10*8000,数据维度，1*10*8000,1表示batch_size,10表示每个特征图的anchor数，8000表示每个anchor的检测结果
// anchor数据是一个一维数组，包含了检测结果的所有信息，每个anchor的数据格式为：[x,y,w,h,confidence,class1,class2,class3,class4,class5]
struct OutputParams
{
	int classtype;   //结果类别，0-4，分别对应5个类别
	float confidence;  //置信度
	//cv::Rect Box;  //检测框
	cv::RotatedRect rotatedBox;  //OBB旋转矩形
	//cv::Mat BoxMask;  //检测框掩码
};
//检查模型路径是否存在
bool CheckModelPath(std::string model_path);
// 为了我们绘制框体，每个类别有自己的颜色框，构造随机颜色函数
void  GetRandomColor(std::vector<cv::Scalar>& colors_Vector, const int& num_class); //num_class是类别数,常量引用
// InputImageConveter函数用于将输入图像缩放到指定大小，同时保持图像比例不变，缩放后的图像用灰色填充，
// 该函数还可以设置是否自动调整图像大小，是否填充灰色，是否放大图像，是否缩小图像，是否设置步长，是否设置颜色
void InputImageConveter(const cv::Mat& image, cv::Mat& OutImage,
	const cv::Size& newShape,
	cv::Vec4d& params,
	bool scaleFill = false,
	bool scaleUp = true,
	const cv::Scalar& color = cv::Scalar(114, 114, 114));
// 画旋转矩形框
void DrawRotatedBox(cv::Mat& img, cv::RotatedRect rotatedBox,cv::Scalar colors ,int thinkness);

// 画检测框,并显示检测结果
void DrawPred(cv::Mat& img,  
			  const std::vector<OutputParams>& outputParams, 
			  std::vector<cv::Scalar>& colors_Vector,
			  const std::vector<std::string>& className
				);
// 检查图像路径并判断是否为图片文件是否存在
bool CheckImagePath(std::string image_path);
