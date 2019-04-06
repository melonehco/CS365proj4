/* arSystem.cpp
 * Reads in camera calibration parameters and uses them to detect a chessboard
 * in video input and project objects onto the video input
 * 
 * to compile:
 * make arSystem
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

void readCalibrationFile(char* calibrationFilename, Mat &cameraMatrix, Mat &distCoeffs)
{
    ifstream paramFile (calibrationFilename);

    if (paramFile.is_open())
    {
        string line;
        char *lineArr = new char[line.length()+1];
        for (int i = 0; i < 3; i++) //loop for 3 rows of camera matrix
        {
            getline(paramFile, line);
            strcpy(lineArr, line.c_str());

            for (int j = 0; j < 3; j++) //loop for 3 columns in each row
            {
                cameraMatrix.at<double>(i, j) = atof( strtok(lineArr, " ") );

            }
        }

        //read in distortion coeffs
        getline(paramFile, line);

        strcpy(lineArr, line.c_str());
        char *word = strtok(lineArr, " ");
        int i = 0;
        while (word != NULL) //loop while there are more words to read in on the line
        {
            distCoeffs.at<double>(i, 0) = atof( word );
            word = strtok(NULL, " ");
            i++;
        }
        
        paramFile.close();
    }
    else
    {
        cout << "unable to open calibration file\n";
        exit(-1);
    }
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
        }
    }
    
    return points;
}

int openVideoInput( Mat cameraMatrix, Mat distCoeffs )
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



    int filenameNum = 0;
    int printIntervalCount = 0;
	for(;;) {
		*capdev >> frame; // get a new frame from the camera, treat as a stream

        vector<Point2f> corner_set;
        vector<Point3f> point_set = buildPointSet(chessboardSize);
        Mat rvec = Mat::zeros(1, 3, DataType<double>::type);
        Mat tvec = Mat::zeros(1, 3, DataType<double>::type);

        bool chessboardFound = findChessboardCorners(frame, chessboardSize, corner_set);

        if (chessboardFound)
        {
            cout << "solving pnp\n";
            solvePnP(point_set, corner_set, cameraMatrix, distCoeffs, rvec, tvec);
            
        }

        imshow("Video", frame);

        //check for user keyboard input
        char key = waitKey(10);
        
		if(key == 'q') {
		    break;
		}

        printIntervalCount++;
        if (printIntervalCount%5 == 0)
        {
            cout << printIntervalCount << "\n";
            cout << chessboardFound << "\n";
            for (int i = 0; i < 3; i++)
            {
                cout << rvec.at<double>(i) << " ";
            }
            cout << "\n";
            for (int i = 0; i < 3; i++)
            {
                cout << tvec.at<double>(i) << " ";
            }
            cout << "\n\n";
        }

	}

	// terminate the video capture
	delete capdev;
    return (0);
}

int main(int argc, char *argv[])
{
    char paramFilename[256];
	// If user didn't give parameter file name
	if(argc < 2) 
	{
		cout << "Usage: ../bin/arSystem |parameter file name|\n";
		exit(-1);
	}

	strcpy(paramFilename, argv[1]);

    Mat cameraMatrix(3, 3, CV_64FC1);
    Mat distCoeffs = Mat::zeros(8, 1, CV_64F);

    //read in camera calibration parameters
    readCalibrationFile(paramFilename, cameraMatrix, distCoeffs);

    openVideoInput(cameraMatrix, distCoeffs);
    


}