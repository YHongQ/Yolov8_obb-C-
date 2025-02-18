#pragma once
#include"yolov8_utils.h"

bool CheckModelPath(std::string model_path)
{
	if (_access(model_path.c_str(), 0) == -1) //access�����ǲ鿴�ļ��Ƿ����,���ڷ���0,�����ڷ���-1. 
											  //c_str()��������һ��ָ������C�ַ�����ָ��, �����뱾string����ͬ.
	{
		std::cout << "Model path does not exist!" << std::endl;
		return false;
	}
	return true;
}
void GetRandomColor(std::vector<cv::Scalar>& colors_Vector, const int& num_class)
{
	/*
	params: colors_Vector ��ɫ����,cv::Scalar��opencv�е���ɫ��,���ڱ�ʾ��ɫ,��һ����ά����
	params: num_class �����
	*/
	// ����ʱ���������������ɫ
	srand(time(0));
	// �̶����ͶȺ�����
	float saturation = 0.7f;  // ���Ͷ� (0.0 �� 1.0)
	float value = 0.9f;        // ���� (0.0 �� 1.0)

	for (int i = 0; i < num_class; i++)
	{
		// ���ȷֲ�ɫ��
		float hue = static_cast<float>(i) / num_class * 360.0f;

		// ��HSVת��ΪBGR
		cv::Mat3b hsv(1, 1, cv::Vec3b(hue, saturation * 255, value * 255));
		cv::Mat3b bgr;
		cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

		// ��ȡBGRֵ
		cv::Vec3b color = bgr(0, 0);
		colors_Vector.push_back(cv::Scalar(color[0], color[1], color[2]));
	}

	std::cout << "GetRandomColor success!" << std::endl;
}

void InputImageConveter(const cv::Mat& image, cv::Mat& OutImage,const cv::Size& newShape,
	cv::Vec4d& params,bool scaleFill ,bool scaleUp,
	const cv::Scalar& color)
{
	/*
	params:image ����ͼ��
	params:OutImage ���ͼ��
	params:newShape �µ�ͼ��ߴ�
	params:params ����,���ڴ洢[ratio_x,ratio_y,dw,dh]
	params:scaleFill �Ƿ�ֱ�ӽ���resizeͼ��
	params:sclaUp �Ƿ�Ŵ�ͼ��
	���ľ���ֱ��reshape�ᵼ��ͼ��ĳ���ȷ����ı䣬������Ҫ����Ŀ����״����ͼ���С��Ҳ����˵����Ҫ����ͼ��ĳ���Ȳ���
	���еȱ������ţ�Ȼ�������ɫ��ͬʱ��֤�ȱ������ŵ�ͼ��λ�������ͼ������λ��
	*/

	//cv::Vec4d��opencv�е�һ����ά������,���ڱ�ʾ��ά����

	cv::Size InputImageShape = image.size();
	float R = std::min((float)newShape.height / (float)InputImageShape.height,
		(float)newShape.width / (float)InputImageShape.width);//�������ű���,ȡ����ȵ���Сֵ,��ΪҪ����ͼ��ĳ���Ȳ���
	
	if (!scaleUp)//�Ƿ����Ŀ����״����ͼ���С�����Ϊfalse,������Ŵ�ͼ��
	{
		R = std::min(R, 1.0f);
	}
	float Ratio[2] = { R,R };//���ű���{ R,R };
	int New_Un_padding[2] = {(int)std::round((float)InputImageShape.width *R),
							 (int)std::round((float)InputImageShape.height * R) };//���ź��ͼ��ߴ�,δ�������
	auto dw = (float)(newShape.width - New_Un_padding[0]);//�������Ŀ��
	auto dh = (float)(newShape.height - New_Un_padding[1]);//�������ĸ߶�
	

	if (scaleFill) //���ѡ��ǿ������ͼ����ȫ����Ŀ��ߴ磨���������������
	{
		dw = 0.0f;
		dh = 0.0f;
		New_Un_padding[0] = newShape.width;
		New_Un_padding[1] = newShape.height;
		Ratio[0] = (float)newShape.width / (float)InputImageShape.width;
		Ratio[1] = (float)newShape.height / (float)InputImageShape.height;
	}
	dw /= 2.0f;//�������Ŀ��
	dh /= 2.0f;//�������ĸ߶�
	// ���������ͼ��
	if (InputImageShape.width != New_Un_padding[0] && InputImageShape.height != New_Un_padding[1])
	{
		cv::resize(image, OutImage, cv::Size(New_Un_padding[0], New_Un_padding[1]));
	}
	else
	{
		OutImage = image.clone();
	}

	int top = int(std::round(dh - 0.1f));//���������ϱ߽�
	int bottom = int(std::round(dh + 0.1f));//���������±߽�
	int left = int(std::round(dw - 0.1f));//����������߽�
	int right = int(std::round(dw + 0.1f));//���������ұ߽�

	params[0] = Ratio[0];//�洢���ű���
	params[1] = Ratio[1];//�洢���ű���
	params[2] = left;//�洢������߽�dw
	params[3] = top;//�洢�����ϱ߽�dh
	cv::copyMakeBorder(OutImage, OutImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);//���ͼ��
}


// ����ת���ο�
void DrawRotatedBox(cv::Mat& img, cv::RotatedRect rotatedBox, cv::Scalar colors, int thinkness)
{
	/*
	params:img ����ͼ��
	params:rotatedBox ��ת���ο�
	params:colors ��ɫ
	params:thinkness �߿�
	*/
	cv::Point2f vertices[4];//��ת���ε��ĸ�����
	rotatedBox.points(vertices);//��ȡ��ת���ε��ĸ�����
	for (int i = 0; i < 4; i++)
	{
		cv::line(img, vertices[i], vertices[(i + 1) % 4], colors, thinkness,8);
	} 
}
// ������,����ʾ�����
void DrawPred(cv::Mat& img,
	const std::vector<OutputParams>& outputParams,
	std::vector<cv::Scalar>& colors_Vector,
	const std::vector<std::string>& className
)
{
	/*
	params:img ����ͼ��
	params:outputParams �������
	params:colors_Vector ��ɫ����
	params:className �������
	*/
	for (int i = 0; i < outputParams.size(); i++)
	{
		int left = 0, top = 0;
		int bottom = 0, right = 0;
		int colorIndex = outputParams[i].classtype;
		//std::cout << colorIndex << std::endl;
		if (outputParams[i].rotatedBox.size.width * outputParams[i].rotatedBox.size.height > 0)
		{
			// ����ת���ο�
			DrawRotatedBox(img, outputParams[i].rotatedBox, colors_Vector[colorIndex], 2);
			//std::cout << "rotatedBox:" << outputParams[i].rotatedBox.size.width << " " << outputParams[i].rotatedBox.size.height << std::endl;
			left = outputParams[i].rotatedBox.center.x; //��ȡ��ת���ο�����ĵ�
			top = outputParams[i].rotatedBox.center.y; //��ȡ��ת���ο�����ĵ�
			bottom = outputParams[i].rotatedBox.size.height / 2 + top;
			right = outputParams[i].rotatedBox.size.width / 2 + left;
			
		}
		//���ŶȺ���������λС��
		std::ostringstream conf_stream;
		conf_stream << std::fixed << std::setprecision(2) << outputParams[i].confidence;
		std::string label = className[outputParams[i].classtype] + ":" + conf_stream.str();
		int baseLine;
		cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.2, 0, &baseLine);
		bottom = MAX(bottom, label_size.height);
		cv::putText(img, label, cv::Point(right, bottom), cv::FONT_HERSHEY_SIMPLEX, 0.6, colors_Vector[colorIndex], 2);
		
	}
	cv::imshow("Detect result", img);
	cv::waitKey(0);
}

// ���ͼ��·���Ƿ����
bool CheckImagePath(std::string image_path)
{
	// �Ƚ����ж�ͼƬ·���Ƿ���ڣ���ͨ���ļ���׺���ж��Ƿ�Ϊͼ���ļ�
	if (_access(image_path.c_str(), 0) == 0)// �ж��ļ��Ƿ����,0��ʾ���ڣ�-1��ʾ������
	{
		std::string image_suffix = image_path.substr(image_path.find_last_of(".") + 1);//��ȡ�ļ���׺��
		std::transform(image_suffix.begin(), image_suffix.end(), image_suffix.begin(), std::tolower);//ת��ΪСд,std::tolowerת��ΪСд,std::tolower�ǻص�����
		if (image_suffix == "jpg" || image_suffix == "jpeg" || image_suffix == "png" || image_suffix == "bmp")
		{
			return true;
		}
		else
		{
			std::cout << "The image suffix is not jpg, jpeg, png, bmp" << std::endl;
			return false;
		}
	}
	std::cout << "The image path is not exist" << std::endl;
	return false;
}