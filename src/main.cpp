/* main.cpp
 * Classifies given image or video input after scanning an 
 * image database first in order to gather data about all object categories.
 * 
 * to run:
 * make main
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

vector<Point2f> detectCorners(Mat imageFrame, Size chessboardSize)
{
    vector<Point2f> corner_set;
    bool chessboardFound = findChessboardCorners(imageFrame, chessboardSize, corner_set);
    
    if (chessboardFound)
    {
        // refine corners
        Mat gray;
        cvtColor(imageFrame, gray, CV_BGR2GRAY);
        Size searchArea(5,5);
        Size zeroZone(-1,-1); // not needed
        TermCriteria criteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001);
        cornerSubPix(gray, corner_set, searchArea, zeroZone, criteria);
    }

    drawChessboardCorners(imageFrame, chessboardSize, corner_set, chessboardFound);
    return corner_set;
}

/**
 * Converts the given corners from pixel coordinates to units of chessboard squares
 */
vector<Point3f> buildPointSet(Size chessboardSize)
{
    vector<Point3f> points;
    
    for (int i = 0; i < chessboardSize.height; i++)
    {
        for (int j = 0; j < chessboardSize.width; j++)
        {
            points.push_back(Point3f(j, -i, 0));
            cout << j << ", " << "-" << i << "\n";
        }
    }
    
    return points;
}

void printCalibrationInfo(Mat cameraMatrix, Mat distCoeffs, double reprojError)
{
    cout << "\ncamera matrix:\n";
    for (int i = 0; i < cameraMatrix.rows; i++)
    {
        for (int j = 0; j < cameraMatrix.cols; j++)
        {
            cout << cameraMatrix.at<double>(i, j) << " ";
        }
        cout << "\n";
    }

    cout << "\ndistortion coefficients:\n";
    for (int i = 0; i < distCoeffs.rows; i++)
    {
        cout << distCoeffs.at<double>(i, 0) << " ";
    }
    cout << "\n";

    cout << "\nre-projection error: " << reprojError << "\n";
}

/**
 * Classifies objects on a live video feed
 */
int openVideoInput( )
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

    Size chessboardSize(9,6); // decided by user

    vector< vector<Point2f> > savedCornerSets;
    vector< vector<Point3f> > savedPointSets;

    double cameraMatrixData[3][3] = 
                            { 
                                {1, 0, frame.cols/2.0},
                                {0, 1, frame.rows/2.0},
                                {0, 0, 1           }
                            };
    Mat cameraMatrix(3,3,CV_64FC1,cameraMatrixData);
    Mat distCoeffs = Mat::zeros(8, 1, CV_64F);
    vector<Mat> rvecs, tvecs;

    int filenameNum = 0;
	for(;;) {
		*capdev >> frame; // get a new frame from the camera, treat as a stream

        vector<Point2f> corners = detectCorners(frame, chessboardSize);

        imshow("Video", frame);
        char key = waitKey(10);
        if(key == 's') {

		    savedCornerSets.push_back(corners);
            savedPointSets.push_back( buildPointSet(chessboardSize) );
            string filename = "calibration_frame_" + to_string(filenameNum) + ".jpg";
            imwrite(filename, frame);
            
            filenameNum++;
		}
        else if(key == 'c') {
            if (savedCornerSets.size() >= 5)
            {
                double reprojError = calibrateCamera(savedPointSets, savedCornerSets, frame.size(), cameraMatrix, distCoeffs, rvecs, tvecs, CV_CALIB_FIX_ASPECT_RATIO);
                printCalibrationInfo(cameraMatrix, distCoeffs, reprojError);
            }
        }
		else if(key == 'q') {
		    break;
		}

	}

	// terminate the video capture
	delete capdev;
    return (0);
}

int main( int argc, char *argv[] ) 
{

    cout << "\nOpening live video..\n";
    openVideoInput();
		
	printf("\nTerminating\n");

	return(0);

}