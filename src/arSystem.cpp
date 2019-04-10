/* arSystem.cpp
 * Reads in camera calibration parameters, uses them to detect a chessboard
 * in video input, and project objects onto the video feed
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

Scalar red = Scalar(0, 0, 255);
Scalar green = Scalar(0, 255, 0);
Scalar blue = Scalar(255, 0, 0);

/**
 * Reads in the given calibration file (in the format written out by calibration.cpp)
 * and writes the camera parameters into the given Mats
 */
void readCalibrationFile(char* calibrationFilename, Mat &cameraMatrix, Mat &distCoeffs)
{
    ifstream paramFile (calibrationFilename);

    if (paramFile.is_open())
    {
        string line;
        char *lineArr = new char[line.length()+1]; //line as char array for strtok

        for (int i = 0; i < 3; i++) //loop for 3 rows of camera matrix
        {
            getline(paramFile, line);
            strcpy(lineArr, line.c_str());

            //for 3 columns in each row
            cameraMatrix.at<double>(i, 0) = atof( strtok(lineArr, " ") );
            cameraMatrix.at<double>(i, 1) = atof( strtok(NULL, " ") ); //NULL b/c continuing on same line
            cameraMatrix.at<double>(i, 2) = atof( strtok(NULL, " ") );
        }

        //read in distortion coeffs
        getline(paramFile, line);

        strcpy(lineArr, line.c_str());
        char *word = strtok(lineArr, " "); //line as char array for strtok
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
}

/**
 * Builds a set of 3D-space points for the corners of a chessboard of the given size,
 * with coordinates in units of chessboard squares
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

/**
 * Projects and draws a set of axes into the given image at the origin,
 * using the given camera parameters and chessboard pose information
 */
void drawAxes(Mat &img, Mat &rvec, Mat &tvec, Mat &cameraMatrix, Mat &distCoeffs)
{
    vector<Point3f> axesPoints{{0, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 0, 1}};
    vector<Point2f> axesImgPoints;
    projectPoints(axesPoints, rvec, tvec, cameraMatrix, distCoeffs, axesImgPoints);
            
    //draw axis lines
    line(img, axesImgPoints[0], axesImgPoints[1], red, 2); //thickness = 2
    line(img, axesImgPoints[0], axesImgPoints[2], green, 2);
    line(img, axesImgPoints[0], axesImgPoints[3], blue, 2);
}

/**
 * Projects and draws a rectangular prism into the given image at the origin,
 * using the given camera parameters and chessboard pose information
 */
void drawRectPrism(Mat &img, Mat &rvec, Mat &tvec, Mat &cameraMatrix, Mat &distCoeffs)
{
    vector<Point3f> points{{0, 0, 0}, {0, 0, 3}, {4, 0, 3}, {4, 0, 0},
                           {0, -1, 0}, {0, -1, 3}, {4, -1, 3}, {4, -1, 0}};
    vector<Point2f> imgPoints;
    projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs, imgPoints);
            
    //draw lines
    line(img, imgPoints[0], imgPoints[1], red, 2); //thickness = 2
    line(img, imgPoints[1], imgPoints[2], red, 2);
    line(img, imgPoints[2], imgPoints[3], red, 2);
    line(img, imgPoints[3], imgPoints[0], red, 2);
    line(img, imgPoints[0], imgPoints[4], green, 2); //thickness = 2
    line(img, imgPoints[1], imgPoints[5], green, 2);
    line(img, imgPoints[2], imgPoints[6], green, 2);
    line(img, imgPoints[3], imgPoints[7], green, 2);
    line(img, imgPoints[4], imgPoints[5], blue, 2); //thickness = 2
    line(img, imgPoints[5], imgPoints[6], blue, 2);
    line(img, imgPoints[6], imgPoints[7], blue, 2);
    line(img, imgPoints[7], imgPoints[4], blue, 2);
}

/**
 * Projects and draws a fish into the given image at the given coordinates,
 * in the given color, using the given camera parameters and chessboard pose
 * information
 */
void drawFish(Mat &img, Scalar &color, float x, float y, Mat &rvec, Mat &tvec, Mat &cameraMatrix, Mat &distCoeffs)
{
    float centerZ = 0.5;
    vector<Point3f> points{//body
                           {x, y, centerZ}, {x + 0.5, y, centerZ + 0.4}, {x + 1.1, y, centerZ}, {x + 0.6, y, centerZ-0.4},
                           //upper fin
                           {x + 0.4, y, centerZ + 0.4}, {x + 0.75, y, centerZ + 0.7}, {x + 1.1, y, centerZ + 0.4}, {x + 0.85, y, centerZ + 0.1},
                           //tail
                           {x + 1.1, y, centerZ}, {x + 1.7, y, centerZ + 0.4}, {x + 1.4, y, centerZ}, {x + 1.7, y, centerZ - 0.4},
                           //lower fin
                           {x + 0.6, y, centerZ - 0.4}, {x + 0.9, y, centerZ - 0.2}, {x + 1.1, y, centerZ - 0.4}, {x + 0.95, y, centerZ - 0.5}
                          };
    vector<Point2f> imgPoints;
    projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs, imgPoints);

    //draw lines
    //body
    line(img, imgPoints[0], imgPoints[1], color, 2); //thickness = 2
    line(img, imgPoints[1], imgPoints[2], color, 2);
    line(img, imgPoints[2], imgPoints[3], color, 2);
    line(img, imgPoints[3], imgPoints[0], color, 2);

    //upper fin
    line(img, imgPoints[4], imgPoints[5], color, 2); //thickness = 2
    line(img, imgPoints[5], imgPoints[6], color, 2);
    line(img, imgPoints[6], imgPoints[7], color, 2);
    line(img, imgPoints[7], imgPoints[4], color, 2);

    //tail
    line(img, imgPoints[8], imgPoints[9], color, 2); //thickness = 2
    line(img, imgPoints[9], imgPoints[10], color, 2);
    line(img, imgPoints[10], imgPoints[11], color, 2);
    line(img, imgPoints[11], imgPoints[8], color, 2);

    //lower fin
    line(img, imgPoints[12], imgPoints[13], color, 2); //thickness = 2
    line(img, imgPoints[13], imgPoints[14], color, 2);
    line(img, imgPoints[14], imgPoints[15], color, 2);
    line(img, imgPoints[15], imgPoints[12], color, 2);
}

/**
 * Project onto a saved image using the given camera parameters
 */
int openImgFile(char* imgName, Mat cameraMatrix, Mat distCoeffs)
{
    cout << "Opening image file " << string(imgName) << "\n";

    // read the image
    Mat src;
    src = imread( string(imgName) );

    // test if the read was successful
    if(src.data == NULL) 
    {
        cout << "Unable to read image" << imgName << "\n";
        exit(-1);
    }

    Size chessboardSize(9,6);

    vector<Point2f> corner_set;
    vector<Point3f> point_set = buildPointSet(chessboardSize);
    Mat rvec = Mat::zeros(1, 3, DataType<double>::type);
    Mat tvec = Mat::zeros(1, 3, DataType<double>::type);

    bool chessboardFound = findChessboardCorners(src, chessboardSize, corner_set);

    if (chessboardFound)
    {
        solvePnP(point_set, corner_set, cameraMatrix, distCoeffs, rvec, tvec);
        
        //drawAxes(src, rvec, tvec, cameraMatrix, distCoeffs);
        //drawRectPrism(src, rvec, tvec, cameraMatrix, distCoeffs);
        drawFish(src, red, 3, 0, rvec, tvec, cameraMatrix, distCoeffs);
        drawFish(src, green, 1, -2, rvec, tvec, cameraMatrix, distCoeffs);
        drawFish(src, blue, 6, -4, rvec, tvec, cameraMatrix, distCoeffs);

        cout << "rvec: ";
        for (int i = 0; i < 3; i++)
        {
            cout << rvec.at<double>(i) << " ";
        }
        cout << "\n";
        cout << "tvec: ";
        for (int i = 0; i < 3; i++)
        {
            cout << tvec.at<double>(i) << " ";
        }
        cout << "\n";
    }

    //display result
    imshow("Image", src);

    //check for user keyboard input
    while (true)
    {
        char key = waitKey(10);
    
        if(key == 'q')
        {
            break;
        }
    }
}

/**
 * Project onto a chessboard inside of precaptured video footage
 */
int openVidFile(const char* vidName, Mat cameraMatrix, Mat distCoeffs)
{
    cout << "Opening video file " << string(vidName) << "\n";
    
    VideoCapture *savedVid = new cv::VideoCapture(vidName);

    // open the video file
	if( !savedVid->isOpened() ) {
		printf("Unable to open video file %s\n", vidName);
		return(-1);
	}

	cv::Size refS( (int) savedVid->get(CAP_PROP_FRAME_WIDTH ),
		       (int) savedVid->get(CAP_PROP_FRAME_HEIGHT));

	printf("Expected size: %d %d\n", refS.width, refS.height);

	namedWindow("Video", 1);
	Mat frame;

    Size chessboardSize(9,6);

    int printIntervalCount = 0;
	for(;;) {
		// read the next frame
        if (savedVid->read(frame) == false)
        {
            cout << "frame empty\n";
            break;            
        }

        vector<Point2f> corner_set;
        vector<Point3f> point_set = buildPointSet(chessboardSize);
        Mat rvec = Mat::zeros(1, 3, DataType<double>::type);
        Mat tvec = Mat::zeros(1, 3, DataType<double>::type);

        bool chessboardFound = findChessboardCorners(frame, chessboardSize, corner_set);

        //project/draw into frame if chessboard found
        if (chessboardFound)
        {
            solvePnP(point_set, corner_set, cameraMatrix, distCoeffs, rvec, tvec);
            
            //drawAxes(frame, rvec, tvec, cameraMatrix, distCoeffs);
            //drawRectPrism(frame, rvec, tvec, cameraMatrix, distCoeffs);
            drawFish(frame, red, 3, 0, rvec, tvec, cameraMatrix, distCoeffs);
            drawFish(frame, green, 1, -2, rvec, tvec, cameraMatrix, distCoeffs);
            drawFish(frame, blue, 6, -4, rvec, tvec, cameraMatrix, distCoeffs);
        }

        imshow("Video", frame);

        //print out rotation and translation vectors every 5 frames
        printIntervalCount++;
        if (printIntervalCount%5 == 0)
        {
            cout << "frame " << printIntervalCount << "\n";
            cout << "rvec: ";
            for (int i = 0; i < 3; i++)
            {
                cout << rvec.at<double>(i) << " ";
            }
            cout << "\n";
            cout << "tvec: ";
            for (int i = 0; i < 3; i++)
            {
                cout << tvec.at<double>(i) << " ";
            }
            cout << "\n\n";
        }

        //check for user keyboard input
        char key = waitKey(10);
		if(key == 'q') {
		    break;
		}
	}

    delete savedVid;

    return (0);
}

/**
 * Looks for chessboard corners on a live video feed and
 * projects onto the video feed with the given parameters if board found
 */
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

    Size chessboardSize(9,6);

    int printIntervalCount = 0;
	for(;;) {
		*capdev >> frame; // get a new frame from the camera, treat as a stream

        vector<Point2f> corner_set;
        vector<Point3f> point_set = buildPointSet(chessboardSize);
        Mat rvec = Mat::zeros(1, 3, DataType<double>::type);
        Mat tvec = Mat::zeros(1, 3, DataType<double>::type);

        bool chessboardFound = findChessboardCorners(frame, chessboardSize, corner_set);

        //project/draw into frame if chessboard found
        if (chessboardFound)
        {
            solvePnP(point_set, corner_set, cameraMatrix, distCoeffs, rvec, tvec);
            
            //drawAxes(frame, rvec, tvec, cameraMatrix, distCoeffs);
            //drawRectPrism(frame, rvec, tvec, cameraMatrix, distCoeffs);
            drawFish(frame, red, 3, 0, rvec, tvec, cameraMatrix, distCoeffs);
            drawFish(frame, green, 1, -2, rvec, tvec, cameraMatrix, distCoeffs);
            drawFish(frame, blue, 6, -4, rvec, tvec, cameraMatrix, distCoeffs);
        }

        imshow("Video", frame);

        //print out rotation and translation vectors every 5 frames
        printIntervalCount++;
        if (printIntervalCount%5 == 0)
        {
            cout << "frame " << printIntervalCount << "\n";
            cout << "rvec: ";
            for (int i = 0; i < 3; i++)
            {
                cout << rvec.at<double>(i) << " ";
            }
            cout << "\n";
            cout << "tvec: ";
            for (int i = 0; i < 3; i++)
            {
                cout << tvec.at<double>(i) << " ";
            }
            cout << "\n\n";
        }

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
    char paramFilename[256];
    char imgOrVidName[256];

	// If user didn't give parameter file name
	if(argc < 2) 
	{
		cout << "Usage: ../bin/arSystem |parameter file name| [Optional image/video file name]\n";
		exit(-1);
	}
    strcpy(paramFilename, argv[1]);

    Mat cameraMatrix(3, 3, CV_64FC1);
    Mat distCoeffs = Mat::zeros(8, 1, CV_64F);

    //read in camera calibration parameters
    readCalibrationFile(paramFilename, cameraMatrix, distCoeffs);
    cout << "Read in calibration file...\n";

    if (argc == 3) //if user gave an image/video filename
    {
        strcpy(imgOrVidName, argv[2]);

        // image
        if( strstr(imgOrVidName, ".jpg") ||
            strstr(imgOrVidName, ".png") ||
            strstr(imgOrVidName, ".ppm") ||
            strstr(imgOrVidName, ".tif") ) 
        {
            openImgFile(imgOrVidName, cameraMatrix, distCoeffs);
        }
        // prerecorded video
        else if (strstr(imgOrVidName, ".mp4") ||
            strstr(imgOrVidName, ".m4v") ||
            strstr(imgOrVidName, ".MOV") ||
            strstr(imgOrVidName, ".mov") ||
            strstr(imgOrVidName, ".avi") )
        {
            openVidFile(imgOrVidName, cameraMatrix, distCoeffs);
        }
        else
        {
            cout << "Not a valid image or video extension\n";
        }
    }
    else // live feed
    {
        openVideoInput(cameraMatrix, distCoeffs);        
    }

    return 0;
}