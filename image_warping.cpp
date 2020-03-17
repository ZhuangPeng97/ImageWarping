#include <iostream>
#include <fstream>
#include <istream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <time.h>

bool ReadLandFromTXT(Eigen::MatrixXf& land, std::string path)
{
	std::ifstream fin(path);
	if (!fin.is_open())
	{
		return false;
	}
	std::string line;
	int i = 0;
	while (fin >> line)
	{
		if (line.empty())
		{
			break;
		}
		std::stringstream ss(line);
		ss >> land(i / 2, i % 2);
		i += 1;
	}
	fin.close();
	return true;
}

bool CalculateCoeff(Eigen::MatrixXf &wv, Eigen::MatrixXf srt_landmark, Eigen::MatrixXf dst_landmark)
{
	int num = srt_landmark.rows();
	Eigen::MatrixXf A= Eigen::MatrixXf::Zero(num + 3, num + 3);
	Eigen::MatrixXf b = Eigen::MatrixXf::Zero(num + 3, 2);
	Eigen::MatrixXf tmp_mat1, tmp_mat2;
	
	for (int i = 0; i < num; i++)
	{	
		int row = num - i;
		tmp_mat1 = dst_landmark.block(i, 0, row, 1).array() - dst_landmark(i, 0);
		tmp_mat2 = dst_landmark.block(i, 1, row, 1).array() - dst_landmark(i, 1);
		A.block(i, i, row, 1) = (tmp_mat1.array().pow(2) + tmp_mat2.array().pow(2)).array().sqrt();
		//tmp_mat = dst_landmark.block(i, 0, row, 2).transpose().colwise() - Eigen::Vector2f(dst_landmark(i, 0), dst_landmark(i, 1));
		//A.block(i, i, row, 1) = (tmp_mat.colwise().norm()).transpose();
		//Eigen::MatrixXf tmp_mat_1 = (A.block(i, i, row, 1));
		tmp_mat1 = A.block(i, i, row, 1);
		A.block(i, i, 1, row) = tmp_mat1.transpose();
		/*
		Eigen::MatrixXf tmp_point = dst_landmark.block(i, 0, 1, 2);
		for (int j = i + 1; j < num; j++)
		{
			A(i, j) = (tmp_point - dst_landmark.block(j, 0, 1, 2)).norm();
			A(j, i) = A(i, j);
		}
		*/
		
	}
	A.block(num, 0, 1, num) = Eigen::MatrixXf::Ones(1, num);
	A.block(0, num, num, 1) = Eigen::MatrixXf::Ones(num, 1);
	A.block(num + 1, 0, 2, num) = dst_landmark.transpose();
	A.block(0, num + 1, num, 2) = dst_landmark;
	b.block(0, 0, num, 2) = srt_landmark - dst_landmark;
	Eigen::ColPivHouseholderQR<Eigen::MatrixXf> qr(A);
	wv.block(0, 0, num + 3, 1) = qr.solve(b.block(0, 0, num+3, 1));
	wv.block(0, 1, num + 3, 1) = qr.solve(b.block(0, 1, num+3, 1));
	return true;
}

bool WarpMapping(cv::Mat& dst, cv::Mat srt, Eigen::MatrixXf dst_landmark, Eigen::MatrixXf wv)
{
	int w = dst.cols;
	int h = dst.rows;
	int num = dst_landmark.rows();
	
	//test1: 4.7s
	Eigen::MatrixXf new_x_mat = Eigen::MatrixXf::Zero(256, 256);
	Eigen::MatrixXf new_y_mat = Eigen::MatrixXf::Zero(256, 256);
	Eigen::MatrixXf base_mat = Eigen::MatrixXf::Zero(256, 256);
	Eigen::MatrixXf tmp_mat = Eigen::MatrixXf::Zero(256, 256);
	Eigen::VectorXf tmp_vec(256);

	clock_t start = clock();
	
	for (int i = 0; i < 256; i++)
	{
		tmp_mat.block(0, i, 256, 1) = i * Eigen::MatrixXf::Ones(256, 1);
		tmp_vec(i) = i;
	}
	clock_t end = clock();
	std::cout << "initialization time: " << double(end - start) / CLOCKS_PER_SEC << std::endl;

	start = clock();
	for (int i = 0; i < num; i++)
	{
		base_mat = (((tmp_mat.array() - dst_landmark(i, 0)).array().pow(2)).colwise() + (tmp_vec.array() - dst_landmark(i, 1)).pow(2)).array().sqrt();
		new_x_mat = new_x_mat + base_mat * wv(i, 0);
		new_y_mat = new_y_mat + base_mat * wv(i, 1);
	}
	new_x_mat = (new_x_mat + wv(num + 2, 0) * tmp_mat.transpose() + (wv(num + 1, 0) + 1) * tmp_mat).array() + wv(num, 0);
	new_y_mat = (new_y_mat + (wv(num + 2, 1) + 1) * tmp_mat.transpose() + wv(num + 1, 1) * tmp_mat).array() + wv(num, 1);
	end = clock();
	std::cout << "interpolation time: " << double(end - start) / CLOCKS_PER_SEC << std::endl;

	float x, y, floor_x, floor_y, u, v;
	cv::Vec3b pix;
	
	start = clock();
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)
		{
			x = new_x_mat(i, j);
			y = new_y_mat(i, j);
			floor_x = floor(x);
			floor_y = floor(y);
			if ((floor_x < 0) || ((floor_x + 1) >= w) || (floor_y < 0) || ((floor_y + 1) >= h)) continue;
			u = x - floor_x;
			v = y - floor_y;
			pix[0] = (1 - u) * (1 - v) * int(srt.at<cv::Vec3b>(floor_y, floor_x)[0]) + u * (1 - v) * int(srt.at<cv::Vec3b>(floor_y, floor_x + 1)[0])
				+ v * (1 - u) * int(srt.at<cv::Vec3b>(floor_y + 1, floor_x)[0]) + u * v * int(srt.at<cv::Vec3b>(floor_y + 1, floor_x + 1)[0]);
			pix[1] = (1 - u) * (1 - v) * int(srt.at<cv::Vec3b>(floor_y, floor_x)[1]) + u * (1 - v) * int(srt.at<cv::Vec3b>(floor_y, floor_x + 1)[1])
				+ v * (1 - u) * int(srt.at<cv::Vec3b>(floor_y + 1, floor_x)[1]) + u * v * int(srt.at<cv::Vec3b>(floor_y + 1, floor_x + 1)[1]);
			pix[2] = (1 - u) * (1 - v) * int(srt.at<cv::Vec3b>(floor_y, floor_x)[2]) + u * (1 - v) * int(srt.at<cv::Vec3b>(floor_y, floor_x + 1)[2])
				+ v * (1 - u) * int(srt.at<cv::Vec3b>(floor_y + 1, floor_x)[2]) + u * v * int(srt.at<cv::Vec3b>(floor_y + 1, floor_x + 1)[2]);
			dst.at<cv::Vec3b>(i, j) = pix;
		}
	}
	end = clock();
	std::cout << "mapping time: " << double(end - start) / CLOCKS_PER_SEC << std::endl;
	return true;
	
	//test2:9.6s
	/*
	//Eigen::MatrixXf tmp_mat = Eigen::MatrixXf::Zero(2, num);
	Eigen::MatrixXf tmp_mat1 = Eigen::MatrixXf::Zero(num, 1);
	Eigen::MatrixXf tmp_mat2 = Eigen::MatrixXf::Zero(num, 1);
	Eigen::MatrixXf base_mat = Eigen::MatrixXf::Ones(1, num + 3);
	float dis_x, dis_y, x, y, u, v;
	int floor_x, floor_y;
	cv::Vec3b pix;
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			//tmp_mat = dst_landmark.transpose().colwise() - Eigen::Vector2f(j, i);	
			//base_mat.block(0, 0, 1, num) = tmp_mat.colwise().norm();
			tmp_mat1 = dst_landmark.block(0, 0, num, 1).array() - j;
			tmp_mat2 = dst_landmark.block(0, 1, num, 1).array() - i;
			base_mat.block(0, 0, 1, num) = (tmp_mat1.array().pow(2) + tmp_mat2.array().pow(2)).array().sqrt().transpose();

			base_mat(0, num + 1) = j;
			base_mat(0, num + 2) = i;
			
			dis_x = (base_mat * wv.block(0, 0, num + 3, 1))(0, 0);
			dis_y = (base_mat * wv.block(0, 1, num + 3, 1))(0, 0);
			x = j + dis_x;
			y = i + dis_y;
			floor_x = floor(x);
			floor_y = floor(y);
			if ((floor_x < 0) || ((floor_x + 1) >= w) || (floor_y < 0) || ((floor_y + 1) >= h)) continue;
			u = x - floor_x;
			v = y - floor_y;
			pix[0] = (1 - u) * (1 - v) * int(srt.at<cv::Vec3b>(floor_y, floor_x)[0]) + u * (1 - v) * int(srt.at<cv::Vec3b>(floor_y, floor_x + 1)[0])
				+ v * (1 - u) * int(srt.at<cv::Vec3b>(floor_y + 1, floor_x)[0]) + u * v * int(srt.at<cv::Vec3b>(floor_y + 1, floor_x + 1)[0]);
			pix[1] = (1 - u) * (1 - v) * int(srt.at<cv::Vec3b>(floor_y, floor_x)[1]) + u * (1 - v) * int(srt.at<cv::Vec3b>(floor_y, floor_x + 1)[1])
				+ v * (1 - u) * int(srt.at<cv::Vec3b>(floor_y + 1, floor_x)[1]) + u * v * int(srt.at<cv::Vec3b>(floor_y + 1, floor_x + 1)[1]);
			pix[2] = (1 - u) * (1 - v) * int(srt.at<cv::Vec3b>(floor_y, floor_x)[2]) + u * (1 - v) * int(srt.at<cv::Vec3b>(floor_y, floor_x + 1)[2])
				+ v * (1 - u) * int(srt.at<cv::Vec3b>(floor_y + 1, floor_x)[2]) + u * v * int(srt.at<cv::Vec3b>(floor_y + 1, floor_x + 1)[2]);
			dst.at<cv::Vec3b>(i, j) = pix;	
		}
	}
	return true;
	*/
}

int main()
{
	std::string srt_img_path = "real_img/1.jpg";
	std::string savepath = "warp_img/1.jpg";
	std::string srt_land_path = "land/1_srt.txt";
	std::string dst_land_path = "land/1_dst.txt";
	int num_ctrl_points = 84; //number of control points
	Eigen::MatrixXf srt_landmark = Eigen::MatrixXf::Zero(84, 2);
	Eigen::MatrixXf dst_landmark = Eigen::MatrixXf::Zero(84, 2);
	if (!ReadLandFromTXT(srt_landmark, srt_land_path))
	{
		std::cout << "read error! srt_img_path does not exist!" << std::endl;
		exit(0);
	}
	if (!ReadLandFromTXT(dst_landmark, dst_land_path))
	{
		std::cout << "read error! dst_img_path does not exist!" << std::endl;
		exit(0);
	}
	
	Eigen::MatrixXf boundary;
	boundary.resize(2, 16);
	boundary << 0, 0, 0, 0, 0, 64, 128, 192, 255, 64, 128, 192, 255, 255, 255, 255,
		0, 64, 128, 192, 255, 255, 255, 255, 255, 0, 0, 0, 0, 64, 128, 192;
	for (int i = 0; i < 16; i++)
	{
		srt_landmark(i + 64, 0) = boundary(0, i);
		srt_landmark(i + 64, 1) = boundary(1, i);
		dst_landmark(i + 64, 0) = boundary(0, i);
		dst_landmark(i + 64, 1) = boundary(1, i);
	}

	Eigen::MatrixXf wv(num_ctrl_points + 3, 2);
	clock_t start = clock();
	CalculateCoeff(wv, srt_landmark, dst_landmark);
	clock_t end = clock();

	std::cout << "CalculateCoeff time: " << double(end - start) / CLOCKS_PER_SEC << std::endl;

	cv::Mat srt_img;
	srt_img = cv::imread(srt_img_path);
	int w = srt_img.cols;
	int h = srt_img.rows;
	cv::Mat dst_img(h, w, CV_8UC3, cv::Scalar(0, 0, 0));

	start = clock();
	WarpMapping(dst_img, srt_img, dst_landmark, wv);
	end = clock();
	cv::imwrite(savepath, dst_img);
	std::cout << "warpmapping time: " << double(end - start) / CLOCKS_PER_SEC << std::endl;
}