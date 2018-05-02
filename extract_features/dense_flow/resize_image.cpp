#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdio.h>
#include <iostream>
using namespace cv;
using namespace cv::gpu;


int main(int argc, char** argv){
	// IO operation
	const char* keys =
		{
			"{ f  | vidFile  | ex2.avi | dir of video frames }"
            "{ n  | frames   | 0       | number of frames }"
			"{ i  | flowFile | flow    | filename of flow image }"
		};

	CommandLineParser cmd(argc, argv, keys);
	string vidFile = cmd.get<string>("vidFile");
	string flowFile = cmd.get<string>("flowFile");
    int frames = cmd.get<int>("frames");

	//VideoCapture capture(vidFile);
	//if(!capture.isOpened()) {
	//	printf("Could not initialize capturing..\n");
	//	return -1;
	//}

    for (int fi = 1 ; fi <= frames ; fi++)
    {   
        char image_file[300];
        sprintf(image_file,"%s/frame-%06d.jpg",vidFile.c_str(),fi);

        Mat image = imread(image_file);

        if (image.empty())
        {   
            fprintf(stderr, "Can't open image [%s]", image_file);
            return -1;
        }

        // resize optical flow
        Mat image_;
        resize(image,image_,cv::Size(340,256));

        char tmp[20];
        sprintf(tmp,"_%04d.jpg",int(fi));
        //imwrite(flowFile + tmp, flow_img_);
        imwrite(flowFile + tmp, image_);
        //frame_num = frame_num + 1;

    }
	return 0;
}
