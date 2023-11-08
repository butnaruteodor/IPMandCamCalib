#pragma once

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>
#include <sys/time.h>
#include <unistd.h>

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "config.h"
#include "calc_arrays.h"

#include "ipm.h"

#include <fstream>

using namespace cv;

int readArray(int *arr, const char *filename);
int saveArray(int *arr, int size, const std::string &filename);
void undistort(uchar3 *inImage, uchar3 *outImage, int *map_x, int *map_y);
void warpImage(uchar3 *inImage, uchar3 *outImage, int *uGrid, int *vGrid);
int loaduvGrid(int uGrid[UV_GRID_COLS], int vGrid[UV_GRID_COLS]);
int loaduvGridFromMatrix(int uGrid[UV_GRID_COLS], int vGrid[UV_GRID_COLS], const MatrixXd &uvGrd);
int loadMappingArrays(int map_x[UV_GRID_COLS], int map_y[UV_GRID_COLS]);
Mat warpImage(Mat inImage, Mat outImage, int uGrid[UV_GRID_COLS], int vGrid[UV_GRID_COLS]);
Mat createEmptyAlphaMat(int rows, int cols);
void toUchar3(Mat frame, uchar3 *output, int width, int height);
void toMat(uchar3 *input, Mat frame, int width, int height);
void equ(int *map_x, int *map_y, int *uGrid, int *vGrid);
VideoCapture getCamera(int camId);