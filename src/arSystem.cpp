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

    //read in camera calibration parameters
    Mat cameraMatrix(3, 3, CV_64FC1);
    Mat distCoeffs = Mat::zeros(8, 1, CV_64F);
    ifstream paramFile ("calibration.txt");
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
        while (word) //loop while there are more words to read in on the line
        {
            distCoeffs.at<double>(i, 0) = atof( word );

            word = strtok(lineArr, " ");
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