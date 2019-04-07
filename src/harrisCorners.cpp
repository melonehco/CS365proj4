/* harrisCorners.cpp
 * 
 * Opens a video feed and detects the Harris corners
 * 
 * to compile:
 * make harrisCorners
 * 
 * Melody Mao & Zena Abulhab
 * CS365 Spring 2019
 * Project 4
 */

#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <ctype.h>
#include <iostream>
#include <fstream> //for writing out to file
#include <iomanip> //for string formatting via a stream
#include <cstring> //for strtok
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace std;
using namespace cv;

void tryDrawHarrisCorners(Mat &imgFrame)
{
    Mat gray;
    cvtColor(imgFrame, gray, CV_BGR2GRAY);

    Mat dst = Mat::zeros( imgFrame.size(), CV_32FC1 );

    int blockSize = 2; // neighborhood size
    int apertureSize = 3; // aka ksize
    double k = .04; // "Harris detector free parameter"
    double thresh = .001;

    cornerHarris( gray, dst, blockSize, apertureSize, k );

    Scalar circleColor = Scalar(0,0,255); // red
    for( int i = 0; i < imgFrame.rows ; i++ )
    {
        for( int j = 0; j < imgFrame.cols; j++ )
        {
            if(dst.at<float>(i,j) > thresh )
            {
                circle( imgFrame, Point(j,i), 5, circleColor, 2, 8, 0 );
            }
        }
    }
}

int openVideoInput()
{
    VideoCapture *capdev;

	// open the video device
	capdev = new cv::VideoCapture(0);
	if( !capdev->isOpened() ) {
		printf("Unable to open video device\n");
		return(-1);
	}

	cv::Size refS( (int) capdev->get(CAP_PROP_FRAME_WIDTH ),
		       (int) capdev->get(CAP_PROP_FRAME_HEIGHT));

	printf("Expected size: %d %d\n", refS.width, refS.height);

	namedWindow("Video", 1);
	Mat frame;

	for(;;) {
		*capdev >> frame; // get a new frame from the camera, treat as a stream
        
        tryDrawHarrisCorners(frame);

        imshow("Video", frame);

        //check for user keyboard input
        char key = waitKey(10);
        
		if(key == 'q') {
		    break;
		}

	}

	// terminate the video capture
	delete capdev;
    return (0);
}

int main(int argc, char *argv[])
{
    openVideoInput();

    return 0;
}