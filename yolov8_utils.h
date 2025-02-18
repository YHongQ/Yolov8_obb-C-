#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>
#include<io.h>
#include<string>
#include<sstream>
#include<iomanip>
#include<time.h>

// ���ڴ洢ģ����������YoLov8-OBB ,input: 1*640*640*3
// output : 1*10*8000,����ά�ȣ�1*10*8000,1��ʾbatch_size,10��ʾÿ������ͼ��anchor����8000��ʾÿ��anchor�ļ����
// anchor������һ��һά���飬�����˼������������Ϣ��ÿ��anchor�����ݸ�ʽΪ��[x,y,w,h,confidence,class1,class2,class3,class4,class5]
struct OutputParams
{
	int classtype;   //������0-4���ֱ��Ӧ5�����
	float confidence;  //���Ŷ�
	//cv::Rect Box;  //����
	cv::RotatedRect rotatedBox;  //OBB��ת����
	//cv::Mat BoxMask;  //��������
};
//���ģ��·���Ƿ����
bool CheckModelPath(std::string model_path);
// Ϊ�����ǻ��ƿ��壬ÿ��������Լ�����ɫ�򣬹��������ɫ����
void  GetRandomColor(std::vector<cv::Scalar>& colors_Vector, const int& num_class); //num_class�������,��������
// InputImageConveter�������ڽ�����ͼ�����ŵ�ָ����С��ͬʱ����ͼ��������䣬���ź��ͼ���û�ɫ��䣬
// �ú��������������Ƿ��Զ�����ͼ���С���Ƿ�����ɫ���Ƿ�Ŵ�ͼ���Ƿ���Сͼ���Ƿ����ò������Ƿ�������ɫ
void InputImageConveter(const cv::Mat& image, cv::Mat& OutImage,
	const cv::Size& newShape,
	cv::Vec4d& params,
	bool scaleFill = false,
	bool scaleUp = true,
	const cv::Scalar& color = cv::Scalar(114, 114, 114));
// ����ת���ο�
void DrawRotatedBox(cv::Mat& img, cv::RotatedRect rotatedBox,cv::Scalar colors ,int thinkness);

// ������,����ʾ�����
void DrawPred(cv::Mat& img,  
			  const std::vector<OutputParams>& outputParams, 
			  std::vector<cv::Scalar>& colors_Vector,
			  const std::vector<std::string>& className
				);
// ���ͼ��·�����ж��Ƿ�ΪͼƬ�ļ��Ƿ����
bool CheckImagePath(std::string image_path);
