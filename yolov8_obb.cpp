#pragma once
#include"yolov8_obb.h"

bool yolov8_obb::ReadModel(cv::dnn::Net& net, const std::string model_path)
{
	/*
	params:net 引用传递，接收模型
	params:model_path 模型路径
	*/
	try //try-catch语句是异常处理的一种方式，用来检测异常并对异常进行处理,
		//这里是检测模型路径是否存在，如果不存在则抛出异常
	{
		if (!CheckModelPath(model_path))
		{
			std::cout << "Model path does not exist!" << std::endl;
			return false;
		}
	}
	catch (const std::exception&) // 捕获异常
	{
		return false;
	}
	net = cv::dnn::readNetFromONNX(model_path); // 从ONNX文件中读取神经网络模型
	// 设置网络的首选后端和目标,首选后端是指在计算机上运行的硬件，首选目标是指在计算机上运行的硬件的类型
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT); // 设置网络的首选后端
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU); // 设置网络的首选目标
	return true;
}

bool yolov8_obb::Detect(cv::Mat& img, std::vector<OutputParams>& outputParams, cv::dnn::Net& net)
{
	/*
	params:img 输入图像
	params:outputParams 输出参数,存储检测结果
	params:net 模型
	*/
	if (img.empty()) //检测图像是否为空
	{
		std::cout << "Image is empty!" << std::endl;
		return false;
	}

	cv::Mat blob; //blob是opencv中的一种数据结构，用于存储图像数据
	outputParams.clear(); //清空输出参数
	int img_col = img.cols; //图像的列数
	int img_row = img.rows; //图像的行数
	cv::Mat netInputImg; //网络输入图像
	cv::Vec4d params; //用于存储[ratio_x,ratio_y,dw,dh]
	InputImageConveter(img, netInputImg, cv::Size(_inputWidth, _inputHeight), params); 
	
	cv::dnn::blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_inputWidth, _inputHeight), cv::Scalar(0, 0, 0), true, false);
	//将图像转换为blob,1/255.0是归一化因子，cv::Size(_inputWidth, _inputHeight)是blob的尺寸，cv::Scalar(0, 0, 0)是均值，true表示交换通道，false表示不翻转
	net.setInput(blob);
	std::vector<cv::Mat> net_output_img;// 这里就是获取网络的输出，实际就是一个三维数组
	net.forward(net_output_img, net.getUnconnectedOutLayersNames());
	//前向传播,获取网络输出,net.getUnconnectedOutLayersNames()获取网络输出层的名称
	//// 输出显示
	//std::cout << net_output_img.size() << std::endl; // 1
	//std::cout << net_output_img[0].size[0] << std::endl; // 1
	//std::cout << net_output_img[0].size[1] << std::endl; //10
	//std::cout << net_output_img[0].size[2] << std::endl; // 8400

	std::vector<int> classType;
	std::vector<float> confidence;
	std::vector<cv::RotatedRect> rotatedBox;
	// 从三维数组中提取出我们想要的数据并对应我们想要的形式。
	cv::Mat output0 = cv::Mat(cv::Size(net_output_img[0].size[2], net_output_img[0].size[1]), 
		CV_32F, (float*)net_output_img[0].data).t();//数据转换，[batch,10,8000]转换为[batch,8000,10]
	//		    将 data 指针强制转换为 float 指针类型
	int net_width = output0.cols; // 网络输出的宽度,就是一个检测目标的信息
	int rows = output0.rows; // 网络输出的行数,候选框的数量
	int class_num = net_width - 5; // 类别数,网络输出的宽度减去5(x,y,w,h,angle)
	int angle_index = net_width - 1; // 角度索引
	float* pdata = (float*)output0.data; // 获取网络输出的数据,指向数据的指针，强制转换为float类型，因为网络输出的数据是float类型
	for (int r = 0; r < rows; r++)
	{
		cv::Mat scores(1, class_num, CV_32FC1, pdata + 4);//获取置信度,pdata+4是因为前四个是x,y,w,h,指针偏移4个位置
		cv::Point classIdPoint; // 类别索引
		double max_class_score; // 最大类别得分
		cv::minMaxLoc(scores, 0, &max_class_score, 0, &classIdPoint); // 获取最大类别得分
		max_class_score = (float)max_class_score;
		// 获取最大类别得分，如果大于类别阈值，则进行后续操作，即认定他是有个有效的检测框
		if (max_class_score >= this->_classThreshold)
		{
			// 获取检测框，并根据我们数据预处理的比例缩放和padding进行还原
			float x = (pdata[0] - params[2]) / params[0]; 
			float y = (pdata[1] - params[3]) / params[1];
			float w = pdata[2] / params[0];
			float h = pdata[3] / params[1];
			float angle = pdata[angle_index] / CV_PI * 180;
			classType.push_back(classIdPoint.x);//类别索引,.x是获取x
			confidence.push_back(max_class_score);//置信度
			rotatedBox.push_back(cv::RotatedRect(cv::Point2f(x, y), cv::Size(w, h), angle));//旋转矩形

		}
		pdata = pdata + net_width; // 指针偏移net_width个位置
	}
	// NMS,非极大值抑制,去除重叠度高的检测框
	std::vector<int> nums_result;//存储NMS后的结果,存储的是检测框的索引
	cv::dnn::NMSBoxes(rotatedBox, confidence, this->_classThreshold, this->_nmsThreshold, nums_result);
	std::vector<std::vector<float>> temp_mask_proposals;//存储检测框掩码信息
	for (int i = 0; i < nums_result.size(); i++)
	{
		int idx = nums_result[i];
		OutputParams result;
		result.classtype = classType[idx];
		result.confidence = confidence[idx];
		result.rotatedBox = rotatedBox[idx];
		// 如果检测框的宽度或者高度小于1，则跳过，因为这样的检测框是无效的
		if (result.rotatedBox.size.width < 1 || result.rotatedBox.size.height < 1)
		{
			continue;
		}
		outputParams.push_back(result); //将检测结果存储到outputParams中
	}
	std::cout << "Detect success!" << std::endl;
	if (outputParams.size())
	{
		return true;
	}
	else
	{
		std::cout << "Not Found something !" << std::endl;
		return false;
	}
	// 测试输出检测结果
	//for (int i = 0; i < outputParams.size(); i++)
	//{
	//	std::cout << "RotatedBox: " << outputParams[i].rotatedBox.center << " " << outputParams[i].rotatedBox.size << " " << outputParams[i].rotatedBox.angle << std::endl;
	//	std::cout << "classType: " << outputParams[i].classtype << " confidence: " << outputParams[i].confidence << std::endl;
	//}

}
	