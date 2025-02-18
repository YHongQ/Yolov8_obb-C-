#pragma once
#include"yolov8_utils.h"

bool CheckModelPath(std::string model_path)
{
	if (_access(model_path.c_str(), 0) == -1) //access函数是查看文件是否存在,存在返回0,不存在返回-1. 
											  //c_str()函数返回一个指向正规C字符串的指针, 内容与本string串相同.
	{
		std::cout << "Model path does not exist!" << std::endl;
		return false;
	}
	return true;
}
void GetRandomColor(std::vector<cv::Scalar>& colors_Vector, const int& num_class)
{
	/*
	params: colors_Vector 颜色向量,cv::Scalar是opencv中的颜色类,用于表示颜色,是一个三维向量
	params: num_class 类别数
	*/
	// 根据时间种子生成随机颜色
	srand(time(0));
	// 固定饱和度和明度
	float saturation = 0.7f;  // 饱和度 (0.0 到 1.0)
	float value = 0.9f;        // 明度 (0.0 到 1.0)

	for (int i = 0; i < num_class; i++)
	{
		// 均匀分布色相
		float hue = static_cast<float>(i) / num_class * 360.0f;

		// 将HSV转换为BGR
		cv::Mat3b hsv(1, 1, cv::Vec3b(hue, saturation * 255, value * 255));
		cv::Mat3b bgr;
		cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

		// 获取BGR值
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
	params:image 输入图像
	params:OutImage 输出图像
	params:newShape 新的图像尺寸
	params:params 参数,用于存储[ratio_x,ratio_y,dw,dh]
	params:scaleFill 是否直接进行resize图像
	params:sclaUp 是否放大图像
	核心就是直接reshape会导致图像的长宽比发生改变，所以需要根据目标形状调整图像大小，也就是说，需要保持图像的长宽比不变
	进行等比例缩放，然后填充颜色，同时保证等比例缩放的图像位于填充后的图像中心位置
	*/

	//cv::Vec4d是opencv中的一个四维向量类,用于表示四维向量

	cv::Size InputImageShape = image.size();
	float R = std::min((float)newShape.height / (float)InputImageShape.height,
		(float)newShape.width / (float)InputImageShape.width);//计算缩放比例,取长宽比的最小值,因为要保持图像的长宽比不变
	
	if (!scaleUp)//是否根据目标形状调整图像大小，如果为false,则不允许放大图像
	{
		R = std::min(R, 1.0f);
	}
	float Ratio[2] = { R,R };//缩放比例{ R,R };
	int New_Un_padding[2] = {(int)std::round((float)InputImageShape.width *R),
							 (int)std::round((float)InputImageShape.height * R) };//缩放后的图像尺寸,未进行填充
	auto dw = (float)(newShape.width - New_Un_padding[0]);//计算填充的宽度
	auto dh = (float)(newShape.height - New_Un_padding[1]);//计算填充的高度
	

	if (scaleFill) //如果选择强制缩放图像完全填满目标尺寸（不保留长宽比例）
	{
		dw = 0.0f;
		dh = 0.0f;
		New_Un_padding[0] = newShape.width;
		New_Un_padding[1] = newShape.height;
		Ratio[0] = (float)newShape.width / (float)InputImageShape.width;
		Ratio[1] = (float)newShape.height / (float)InputImageShape.height;
	}
	dw /= 2.0f;//计算填充的宽度
	dh /= 2.0f;//计算填充的高度
	// 生成填充后的图像
	if (InputImageShape.width != New_Un_padding[0] && InputImageShape.height != New_Un_padding[1])
	{
		cv::resize(image, OutImage, cv::Size(New_Un_padding[0], New_Un_padding[1]));
	}
	else
	{
		OutImage = image.clone();
	}

	int top = int(std::round(dh - 0.1f));//计算填充的上边界
	int bottom = int(std::round(dh + 0.1f));//计算填充的下边界
	int left = int(std::round(dw - 0.1f));//计算填充的左边界
	int right = int(std::round(dw + 0.1f));//计算填充的右边界

	params[0] = Ratio[0];//存储缩放比例
	params[1] = Ratio[1];//存储缩放比例
	params[2] = left;//存储填充的左边界dw
	params[3] = top;//存储填充的上边界dh
	cv::copyMakeBorder(OutImage, OutImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);//填充图像
}


// 画旋转矩形框
void DrawRotatedBox(cv::Mat& img, cv::RotatedRect rotatedBox, cv::Scalar colors, int thinkness)
{
	/*
	params:img 输入图像
	params:rotatedBox 旋转矩形框
	params:colors 颜色
	params:thinkness 线宽
	*/
	cv::Point2f vertices[4];//旋转矩形的四个顶点
	rotatedBox.points(vertices);//获取旋转矩形的四个顶点
	for (int i = 0; i < 4; i++)
	{
		cv::line(img, vertices[i], vertices[(i + 1) % 4], colors, thinkness,8);
	} 
}
// 画检测框,并显示检测结果
void DrawPred(cv::Mat& img,
	const std::vector<OutputParams>& outputParams,
	std::vector<cv::Scalar>& colors_Vector,
	const std::vector<std::string>& className
)
{
	/*
	params:img 输入图像
	params:outputParams 输出参数
	params:colors_Vector 颜色向量
	params:className 类别名称
	*/
	for (int i = 0; i < outputParams.size(); i++)
	{
		int left = 0, top = 0;
		int bottom = 0, right = 0;
		int colorIndex = outputParams[i].classtype;
		//std::cout << colorIndex << std::endl;
		if (outputParams[i].rotatedBox.size.width * outputParams[i].rotatedBox.size.height > 0)
		{
			// 画旋转矩形框
			DrawRotatedBox(img, outputParams[i].rotatedBox, colors_Vector[colorIndex], 2);
			//std::cout << "rotatedBox:" << outputParams[i].rotatedBox.size.width << " " << outputParams[i].rotatedBox.size.height << std::endl;
			left = outputParams[i].rotatedBox.center.x; //获取旋转矩形框的中心点
			top = outputParams[i].rotatedBox.center.y; //获取旋转矩形框的中心点
			bottom = outputParams[i].rotatedBox.size.height / 2 + top;
			right = outputParams[i].rotatedBox.size.width / 2 + left;
			
		}
		//置信度后处理，保留两位小数
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

// 检查图像路径是否存在
bool CheckImagePath(std::string image_path)
{
	// 先进行判断图片路径是否存在，在通过文件后缀名判断是否为图像文件
	if (_access(image_path.c_str(), 0) == 0)// 判断文件是否存在,0表示存在，-1表示不存在
	{
		std::string image_suffix = image_path.substr(image_path.find_last_of(".") + 1);//获取文件后缀名
		std::transform(image_suffix.begin(), image_suffix.end(), image_suffix.begin(), std::tolower);//转换为小写,std::tolower转换为小写,std::tolower是回调函数
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