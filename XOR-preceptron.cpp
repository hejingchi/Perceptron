#include<iostream>
#include<opencv2/opencv.hpp>
#include<iomanip>
#include<cmath>

using namespace std;
using namespace cv;
Mat ReLU(const Mat& h)
{
	Mat result(h.rows, h.cols, CV_32F);
	for (int i = 0; i < h.rows; i++)
	{
		float* p = result.ptr<float>(i);
		for (int j = 0; j < h.cols; j++)
		{
			float f = h.at<float>(i, j);
			if (f > 0)
			{
				p[j] = f;
			}
			else
			{
				p[j] = 0;
			}
		}
	}
	return result;
}
Mat ReLU_derivative(const Mat& h)
{
	Mat result;
	for (int i = 0; i < h.rows; i++)
	{
		float* p = result.ptr<float>(i);
		for (int j = 0; j < h.cols; j++)
		{
			float f = h.at<float>(i, j);
			if (f > 0)
			{
				p[j] = 1;
			}
			else
			{
				p[j] = 0;
			}
		}
	}
	return result;
}
int main_XOR()
{
	//给矩阵直接赋值的方式: Mat mat = ( Mat_<type>(rows, cols) << 1,2,3,... )
	Mat X = (Mat_<float>(4, 2) << 0, 0, 0, 1, 1, 0, 1, 1);
	// 对应感知机里面的四个数据 x[1] x[2] x[3] x[4]
	Mat Y = (Mat_<float>(4, 1) << 1, 0, 0, 1);
	//X 矩阵存储数据点， Y矩阵（向量）存储数据的类别
	Mat W1(2, 6, CV_32F); randn(W1, 0, 1);
	Mat b1 = Mat::ones(1, 6, CV_32F);
	Mat W2(6, 1, CV_32F); randn(W2, 0, 1);
	float b2 = 0;
	float lr = 0.3;//学习率
	// 具体解释
	// 参数的个数对应W1矩阵的列 原始向量x[i]的维度对应W1矩阵的行
	// x是二位行向量，所以W1是两行的 W1需要生成6个参数所以W1是6列的
	// W1 * x + b1 得到的是一个1 * 6 的行向量 再经过ReLU进行归一化操作 去除无用信息
	// 现在的目的就是对这个1 * 6的行向量，再进行第二层运算，第二层就是普通的感知机过程
	// 因此这里是是6维行向量，所以需要W2是2行的 由于最后是直接输出结果了，和感知机一样
	// 所以这里的矩阵是1列的 对应的b2也是一维的
	// 即 W2 * h + b2, h = ReLU( W1 * x + b1)
	// 最后再用sigmoid返回结果，或者直接就是返回原始值
	//
	for (int e = 0; e < 10000; e++)
	{
		float error = 0;
		for (int i = 0; i < 4; i++)
		{
			double error = 0;
			Mat x = X.row(i);//读取当前学习的x参数，1 * 2
			Mat h = W1 * x + b1;//计算第一层的矩阵惩罚 1 * 6
			h = ReLU(h);//h是1 * 6 的行向量
			float y = Mat(h * W2).at<float>(0, 0) + b2;
			double err = y - Y.at<float>(i);//计算当前误差 误差就是和目标值Y.at<float>(i)的距离的平方
			error += pow(err, 2);
			//err是y对应的误差 我们要作的是对W2，W1，b2，b1从这个err参数里面求偏导
			W2 -= lr * h.t() * err;//W2矩阵更新
			b2 -= lr * err;
			W1 -= lr * x.t() * W2.t().mul(ReLU_derivative(h));
			b1 -= lr * err * W2.t().mul(ReLU_derivative(h));
		}
		
	}
	// 4. 测试模型
	cout << "\n--- 测试结果 ---" << endl;
	for (int i = 0; i < 4; i++) {
		Mat h = ReLU(X.row(i) * W1 + b1);
		float res = Mat((h * W2 + b2)).at<float>(0, 0);
		cout << "输入: " << X.row(i) << " 预测值: " << res << " (目标: " << Y.at<float>(i) << ")" << endl;
	}

	return 0;

}
