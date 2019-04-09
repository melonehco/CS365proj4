/* extension2.cpp
 * TODO:
 * 
 * to compile:
 * make extension2
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
#include <numeric>
#include <ctype.h>
#include <iostream>
#include <fstream> //for writing out to file
#include <iomanip> //for string formatting via a stream
#include <cstring> //for strtok
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <GL/gl.h>

using namespace std;
using namespace cv;

Scalar red = Scalar(0, 0, 255);
Scalar green = Scalar(0, 255, 0);
Scalar blue = Scalar(255, 0, 0);

GLuint texture;

/* Loads the OpenGL texture
 */
int loadTexture()
{
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    //make red Mat
    Mat redImg(600, 400, CV_8UC3, Scalar(0,0,255));

    //TODO: get texture width and height
    //[keep], mipmap level, format, width, height, [keep], src format, src type, img data
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, redImg.cols, redImg.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, redImg.data);

    return 0;
}

/* basically glutSolidCube but with texture coords
 * modified from https://stackoverflow.com/questions/327043/how-to-apply-texture-to-glutsolidcube */
void solidCube(GLdouble size)
{
  GLfloat n[6][3] =
  {
    {-1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0},
    {1.0, 0.0, 0.0},
    {0.0, -1.0, 0.0},
    {0.0, 0.0, 1.0},
    {0.0, 0.0, -1.0}
  };
    GLint faces[6][4] =
  {
    {0, 1, 2, 3},
    {3, 2, 6, 7},
    {7, 6, 5, 4},
    {4, 5, 1, 0},
    {5, 6, 2, 1},
    {7, 4, 0, 3}
  };
  GLfloat v[8][3];
  GLint i;

  v[0][0] = v[1][0] = v[2][0] = v[3][0] = -size / 2;
  v[4][0] = v[5][0] = v[6][0] = v[7][0] = size / 2;
  v[0][1] = v[1][1] = v[4][1] = v[5][1] = -size / 2;
  v[2][1] = v[3][1] = v[6][1] = v[7][1] = size / 2;
  v[0][2] = v[3][2] = v[4][2] = v[7][2] = -size / 2;
  v[1][2] = v[2][2] = v[5][2] = v[6][2] = size / 2;

  //added
    GLfloat tc[4][2] =
    {
        { 0.0, 0.0 },
        { 1.0, 0.0 },
        { 1.0, 1.0 },
        { 0.0, 1.0 }
    };
  //

  for (i = 5; i >= 0; i--) {
    glBegin(GL_QUADS);
    glNormal3fv(&n[i][0]);

    glTexCoord2fv(&tc[0][0]);
    glVertex3fv(&v[faces[i][0]][0]);
    glTexCoord2fv(&tc[1][0]);
    glVertex3fv(&v[faces[i][1]][0]);
    glTexCoord2fv(&tc[2][0]);
    glVertex3fv(&v[faces[i][2]][0]);
    glTexCoord2fv(&tc[3][0]);
    glVertex3fv(&v[faces[i][3]][0]);
    glEnd();
  }
}

/* Callback function to draw with OpenGL on each frame
 */
void drawOpenGL(void *params)
{
    glLoadIdentity();
    glBindTexture(GL_TEXTURE_2D, texture);

    //TODO: get rvec & tvec, use w/ glRotatef, glTranslatef

    solidCube(1.0);
}

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

            //for 3 columns in each row
            cameraMatrix.at<double>(i, 0) = atof( strtok(lineArr, " ") );
            cameraMatrix.at<double>(i, 1) = atof( strtok(NULL, " ") );
            cameraMatrix.at<double>(i, 2) = atof( strtok(NULL, " ") );
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

    string ugh;
    cin >> ugh;
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

    const string winName = "Video";
	namedWindow(winName, WINDOW_OPENGL); //OpenGL tag
	Mat frame;

    Size chessboardSize(9,6);

    //for OpenGl textures
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &texture);
    loadTexture();

    setOpenGlDrawCallback(winName, drawOpenGL);

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
            solvePnP(point_set, corner_set, cameraMatrix, distCoeffs, rvec, tvec);
            
            //drawAxes(frame, rvec, tvec, cameraMatrix, distCoeffs);
            //drawRectPrism(frame, rvec, tvec, cameraMatrix, distCoeffs);
            drawFish(frame, red, 3, 0, rvec, tvec, cameraMatrix, distCoeffs);
            drawFish(frame, green, 1, -2, rvec, tvec, cameraMatrix, distCoeffs);
            drawFish(frame, blue, 6, -4, rvec, tvec, cameraMatrix, distCoeffs);
        }

        imshow(winName, frame);
        updateWindow(winName);

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

    // if (argc == 3) 
    // {
    //     strcpy(imgOrVidName, argv[2]);

    //     // image
    //     if( strstr(imgOrVidName, ".jpg") ||
    //         strstr(imgOrVidName, ".png") ||
    //         strstr(imgOrVidName, ".ppm") ||
    //         strstr(imgOrVidName, ".tif") ) 
    //     {
    //         openImgFile(imgOrVidName, cameraMatrix, distCoeffs);
    //     }
    //     // prerecorded video
    //     else if (strstr(imgOrVidName, ".mp4") ||
    //         strstr(imgOrVidName, ".m4v") ||
    //         strstr(imgOrVidName, ".MOV") ||
    //         strstr(imgOrVidName, ".mov") ||
    //         strstr(imgOrVidName, ".avi") )
    //     {
    //         openVidFile(imgOrVidName, cameraMatrix, distCoeffs);
    //     }
    //     else
    //     {
    //         cout << "Not a valid image or video extension\n";
    //     }
    // }
    // else // live feed
    {
        openVideoInput(cameraMatrix, distCoeffs);        
    }

    return 0;
}
