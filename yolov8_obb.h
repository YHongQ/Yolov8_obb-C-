#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>
#include"yolov8_utils.h"

class yolov8_obb
{
public:
	yolov8_obb() {};
	~yolov8_obb() {};
	// ��ȡģ��
	bool ReadModel(cv::dnn::Net& net, const std::string model_path);
	// ���OBB����
	bool Detect(cv::Mat& img, std::vector<OutputParams>& outputParams, cv::dnn::Net &net);
	int _inputWidth = 640;//ONNXģ������Ŀ��
	int _inputHeight = 640;//ONNXģ������ĸ߶�
	std::vector<std::string> _classes = {"car","bus","truck","feright_car","van"};//�������
	// 0:car, 1:bus, 2:truck, 3:feright_car, 4:van .��Ӧ5����� 
	int _num_classes = _classes.size();//�����

private: 
	float _classThreshold = 0.25;//�����ֵ,�������˵�����ֵ�����
	float _nmsThreshold = 0.45;//NMS��ֵ,���������ص��ȸߵļ���
};
