#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"

#include <stdio.h>
#include <iostream>
using namespace cv;
using namespace cv::gpu;

//static void convertFlowToImage(const Mat &flow_x, const Mat &flow_y, Mat &img,
//       double lowerBound, double higherBound) {
//	#define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(v))
//	for (int i = 0; i < flow_x.rows; ++i) {
//		for (int j = 0; j < flow_y.cols; ++j) {
//			float x = flow_x.at<float>(i,j);
//  		float y = flow_y.at<float>(i,j);
//          float m = cvSqrt(x*x + y*y);
//          float scale = 128 / higherBound;
//			img.at<Vec3b>(i,j)[2] = CAST(x*scale+128, 0, 255);
//			img.at<Vec3b>(i,j)[1] = CAST(y*scale+128, 0, 255);
//          img.at<Vec3b>(i,j)[0] = CAST(m*scale+128, 0, 255);
//        }
//	}
//	#undef CAST
//}

static void convertFlowToImage(const Mat &flow_x, const Mat &flow_y, Mat &img,
       double lowerBound, double higherBound) {
    #define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
    for (int i = 0; i < flow_x.rows; ++i) {
        for (int j = 0; j < flow_y.cols; ++j) {
            float x = flow_x.at<float>(i,j);
            float y = flow_y.at<float>(i,j);
            float m = cvSqrt(x*x + y*y);
            img.at<Vec3b>(i,j)[2] = CAST(x, lowerBound, higherBound);
            img.at<Vec3b>(i,j)[1] = CAST(y, lowerBound, higherBound);
            img.at<Vec3b>(i,j)[0] = CAST(m, lowerBound, higherBound);
        }
    }
    #undef CAST
}

//static void convertFlowToImage(const Mat &flow_x, const Mat &flow_y, Mat &img_x, Mat &img_y,
//       double lowerBound, double higherBound) {
//    #define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
//    for (int i = 0; i < flow_x.rows; ++i) {
//        for (int j = 0; j < flow_y.cols; ++j) {
//            float x = flow_x.at<float>(i,j);
//            float y = flow_y.at<float>(i,j);
//            img_x.at<uchar>(i,j) = CAST(x, lowerBound, higherBound);
//            img_y.at<uchar>(i,j) = CAST(y, lowerBound, higherBound);
//        }
//    }
//    #undef CAST
//}

static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,double, const Scalar& color){
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(x,y), 2, color, -1);
        }
}

int main(int argc, char** argv){
	// IO operation
	const char* keys =
		{
			"{ f  | vidFile | ex2.avi | dir of video frames }"
            "{ n  | frames  | 0       | number of frames }"
			"{ i  | flowFile | flow   | filename of flow image }"
			"{ b  | bound | 15 | specify the maximum of optical flow }"
			"{ t  | type | 0 | specify the optical flow algorithm }"
			"{ d  | device_id    | 0  | set gpu id }"
			"{ s  | step  | 1 | specify the step for frame sampling }"
		};

	CommandLineParser cmd(argc, argv, keys);
	string vidFile = cmd.get<string>("vidFile");
	string flowFile = cmd.get<string>("flowFile");
    int frames = cmd.get<int>("frames");
	int bound = cmd.get<int>("bound");
    int type  = cmd.get<int>("type");
    int device_id = cmd.get<int>("device_id");
    int step = cmd.get<int>("step");

	//VideoCapture capture(vidFile);
	//if(!capture.isOpened()) {
	//	printf("Could not initialize capturing..\n");
	//	return -1;
	//}

	//int frame_num = 0;
	//Mat image, prev_image, prev_grey, grey, frame, flow_x, flow_y;
    Mat flow_x, flow_y;
	GpuMat frame_0, frame_1, flow_u, flow_v;

	setDevice(device_id);
	FarnebackOpticalFlow alg_farn;
	OpticalFlowDual_TVL1_GPU alg_tvl1;
	BroxOpticalFlow alg_brox(0.197f, 50.0f, 0.8f, 10, 77, 10);

    int skip = step;
    for (int fi = 1 ; fi < frames ; fi++)
    {   
        char image_file_0[300];
        char image_file_1[300];
        sprintf(image_file_0,"%s/frame-%06d.jpg",vidFile.c_str(),fi);
        sprintf(image_file_1,"%s/frame-%06d.jpg",vidFile.c_str(),fi+skip);

        Mat image_0 = imread(image_file_0);
        Mat image_1 = imread(image_file_1);

        if (image_0.empty())
        {   
            fprintf(stderr, "Can't open image [%s]", image_file_0);
            return -1;
        }
        if (image_1.empty())
        {
            fprintf(stderr, "Can't open image [%s]", image_file_1);
            return -1;
        }

        if (image_0.size() != image_1.size())
        {
            fprintf(stderr, "Images should be of equal sizes");
            return -1;
        }

        Mat image_0_gray(image_0.size(), CV_8UC1);
        Mat image_1_gray(image_1.size(), CV_8UC1);
        cvtColor(image_0, image_0_gray, CV_BGR2GRAY);
        cvtColor(image_1, image_1_gray, CV_BGR2GRAY);

        // GPU mat
        frame_0.upload(image_0_gray);
        frame_1.upload(image_1_gray);

        // GPU optical flow
        switch(type){
        case 0:
            alg_farn(frame_0,frame_1,flow_u,flow_v);
            break;
        case 1:
            alg_tvl1(frame_0,frame_1,flow_u,flow_v);
            break;
        case 2:
            GpuMat d_frame0f, d_frame1f;
            frame_0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
            frame_1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);
            alg_brox(d_frame0f, d_frame1f, flow_u, flow_v);
            break;
        }

        flow_u.download(flow_x);
        flow_v.download(flow_y);

        // Output optical flow
        Mat flow_img(flow_x.size(),CV_8UC3);
        convertFlowToImage(flow_x, flow_y, flow_img, -bound, bound);
        char tmp[20];
        sprintf(tmp,"-%06d.jpg",int(fi));

        //Mat flow_img_;
        //resize(flow_img,flow_img_,cv::Size(340,256));

        //imwrite(flowFile + tmp, flow_img_);
        imwrite(flowFile + tmp, flow_img);
        //frame_num = frame_num + 1;

    }
	return 0;
}
