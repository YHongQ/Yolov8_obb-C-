#include<iostream>
#include<vector>
#include<math.h>
#include<time.h>
#include<opencv2/opencv.hpp>
#include<string>
#include"yolov8_obb.h"
#include"yolov8_utils.h"
#include<chrono>

using namespace std;
using namespace cv;
using namespace cv::dnn;

template<class T>
void Yolov8(T& task, cv::Mat& img, std::string& model_path)
{
	cv::imshow("Input Image", img);
	cv::dnn::Net net;// ��ʼ������
	if (task.ReadModel(net, model_path)) // ��ȡģ��
	{
		std::cout << "Model loaded successfully!" << std::endl;
	}
	else
	{
		std::cout << "Model loaded failed!" << std::endl;
		return;
	}
	vector<cv::Scalar> colors_Vector;
	GetRandomColor(colors_Vector, task._num_classes); // Ϊÿ��������������ɫ
	vector<OutputParams> outputParams;
	auto begin = chrono::high_resolution_clock::now();
	if (task.Detect(img, outputParams, net))
	{
		auto end = chrono::high_resolution_clock::now();
		chrono::duration<double, milli> detect_time = end - begin;
		std::cout << "Inference time : " << detect_time.count() <<" ms"<< std::endl;
		DrawPred(img, outputParams, colors_Vector, task._classes);
	}
	else
	{
		return;
	}
}

int main(int argc, char** argv)
{
	std::string filename = "D:/CODE/MyUndergraduateDesign/Images/00125.jpg";
	std::string model_path = "D:/CODE/MyUndergraduateDesign/Models/best_640640_10_8400.onnx";

	if (argc >= 3)
	{
		// �ж����������������һ����ģ��·������һ����ͼƬ·��
		if (CheckImagePath(argv[1]))
		{
			filename = argv[1];
			model_path = argv[2];
		}
		else if (CheckImagePath(argv[2]))
		{
			model_path = argv[1];
			filename = argv[2];
		}
		else
		{
			cout << "Please input the correct path!" << endl;
			return 0;
		}
	}
	// ��ӡʹ�õ�·��
	cout << "Using image: " << filename << endl;
	cout << "Using model: " << model_path << endl;
	yolov8_obb obb_opencv;
	cv::Mat srcimg = cv::imread(filename);
	cv::Mat img = srcimg.clone();
	// ģ���ʹ�ã���ʽָ��ģ�����
	Yolov8<yolov8_obb>(obb_opencv, img, model_path);
	cout << "---------------------------------------------------" << endl;
	system("pause");
	return 0;	
}