#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>
#include"yolov8_utils.h"

class yolov8_obb
{
public:
	yolov8_obb() {};
	~yolov8_obb() {};
	// 读取模型
	bool ReadModel(cv::dnn::Net& net, const std::string model_path);
	// 检测OBB框体
	bool Detect(cv::Mat& img, std::vector<OutputParams>& outputParams, cv::dnn::Net &net);
	int _inputWidth = 640;//ONNX模型输入的宽度
	int _inputHeight = 640;//ONNX模型输入的高度
	std::vector<std::string> _classes = {"car","bus","truck","feright_car","van"};//类别名称
	// 0:car, 1:bus, 2:truck, 3:feright_car, 4:van .对应5个类别 
	int _num_classes = _classes.size();//类别数

private: 
	float _classThreshold = 0.25;//类别阈值,用来过滤低于阈值的类别
	float _nmsThreshold = 0.45;//NMS阈值,用来过滤重叠度高的检测框
};
