#include <iostream>
#include <fstream>
#include <omp.h>
#include <sys/time.h>
#include <unistd.h>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"

#include "ipm.h"
#include "calc_arrays.h"
#include "config.h"

using namespace std;
using namespace cv;

#define UV_GRID_ROWS 2
#define UV_GRID_COLS 524288

#define OUT_IMAGE_WIDTH 1024
#define OUT_IMAGE_HEIGHT 512
uchar3 *aux;
Mat outImage(OUT_IMAGE_HEIGHT, OUT_IMAGE_WIDTH, CV_8UC3, Scalar(0, 0, 0));

std::string gstreamer_pipeline(int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method)
{
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true sync=false";
}

int readArray(int *arr, const char *filename);
void undistort(uchar3 *inImage, uchar3 *outImage, int *map_x, int *map_y);
void warpImage(uchar3 *inImage, uchar3 *outImage, int *uGrid, int *vGrid);
int loaduvGrid(int uGrid[UV_GRID_COLS], int vGrid[UV_GRID_COLS]);
int loaduvGridFromMatrix(int uGrid[UV_GRID_COLS], int vGrid[UV_GRID_COLS], const MatrixXd& uvGrd);
int loadMappingArrays(int map_x[UV_GRID_COLS], int map_y[UV_GRID_COLS]);
Mat warpImage(Mat inImage, int uGrid[UV_GRID_COLS], int vGrid[UV_GRID_COLS]);
Mat createEmptyAlphaMat(int rows, int cols);
void toUchar3(Mat frame, uchar3 *output, int width, int height);
void toMat(uchar3 *input, Mat frame, int width, int height);
void equ(int *map_x, int *map_y, int *uGrid, int *vGrid);
int main()
{
    uchar3 *input, *output, *output_calib;
    int *uGrid, *vGrid;
    int *uGridCpu = new int[UV_GRID_COLS];
    int *vGridCpu = new int[UV_GRID_COLS];

    cudaError_t err_x = cudaMallocManaged((void **)&input, 1920 * 1080 * sizeof(uchar3));
    cudaError_t err_y = cudaMallocManaged((void **)&output, 1024 * 512 * sizeof(uchar3));
    cudaError_t err_y1 = cudaMallocManaged((void **)&output_calib, 1920 * 1080 * sizeof(uchar3));
    cudaError_t err_y2 = cudaMallocManaged((void **)&aux, 1920 * 1080 * sizeof(uchar3));

    // // Check for errors
    if (err_x != cudaSuccess || err_y != cudaSuccess /*|| err_output != cudaSuccess || err_output2 != cudaSuccess*/)
    {
        fprintf(stderr, "Failed to allocate unified memory\n");
        return 1; // or handle error appropriately
    }

        // Load uv grid from file
    //loadMappingArrays(uGridCpu, vGridCpu);


    CameraInfo cameraInfo;
    IpmInfo ipmInfo;

    cameraInfo.focalLengthX = (int)(IN_IMAGE_WIDTH/2.75)*6.45;
    cameraInfo.focalLengthY = (int)(IN_IMAGE_HEIGHT/2.75)*3.63;
    cameraInfo.opticalCenterX = IN_IMAGE_WIDTH/2;
    cameraInfo.opticalCenterY = IN_IMAGE_HEIGHT/2;
    cameraInfo.cameraHeight = 267;
    cameraInfo.pitch=25.0;
    cameraInfo.yaw = 0;
    cameraInfo.roll = 0;

    ipmInfo.inputWidth = IN_IMAGE_WIDTH;
    ipmInfo.inputHeight = IN_IMAGE_HEIGHT;
    ipmInfo.left = 5;
    ipmInfo.right = IN_IMAGE_WIDTH - 5;
    ipmInfo.top = 200;
    ipmInfo.bottom = IN_IMAGE_HEIGHT-190;

    MatrixXd uvGrd = GetMappingArrays(cameraInfo,ipmInfo);

    loaduvGridFromMatrix(uGridCpu,vGridCpu,uvGrd);

    cudaMalloc((void **)&uGrid, UV_GRID_COLS * sizeof(int));
    cudaMemcpy(uGrid, uGridCpu, UV_GRID_COLS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&vGrid, UV_GRID_COLS * sizeof(int));
    cudaMemcpy(vGrid, vGridCpu, UV_GRID_COLS * sizeof(int), cudaMemcpyHostToDevice);

    Mat frame;
    Mat outFrame;
    //--- INITIALIZE VIDEOCAPTURE
    int capture_width = 1920;
    int capture_height = 1080;
    int display_width = 1920;
    int display_height = 1080;
    int framerate = 30;
    int flip_method = 2;

    std::string pipeline = gstreamer_pipeline(capture_width,
                                              capture_height,
                                              display_width,
                                              display_height,
                                              framerate,
                                              flip_method);

    VideoCapture cap(pipeline, CAP_GSTREAMER);
    
    if (!cap.isOpened())
    {
        std::cout << "Failed to open camera." << std::endl;
        return (-1);
    }
    cv::namedWindow("CSI Camera", cv::WINDOW_AUTOSIZE);

    std::cout << "Hit ESC to exit"
              << "\n";
    while (true)
    {
        if (!cap.read(frame))
        {
            std::cout << "Capture read error" << std::endl;
            break;
        }
        toUchar3(frame, input, frame.cols, frame.rows);
        warpImageK(input,output,uGrid,vGrid);
        cudaDeviceSynchronize();
        toMat(output, outImage, OUT_IMAGE_WIDTH, OUT_IMAGE_HEIGHT);

        cv::imshow("CSI Camera", outImage);
        int keycode = cv::waitKey(10) & 0xff;
        if (keycode == 27)
            break;
    }

    cap.release();
    cv::destroyAllWindows();

    cudaFree(input);
    cudaFree(output);
    cudaFree(aux);
    cudaFree(output_calib);
    cudaFree(uGrid);
    cudaFree(vGrid);
    delete[] uGridCpu;
    delete[] vGridCpu;
    return 0;
}

Mat warpImage(Mat inImage, int uGrid[UV_GRID_COLS], int vGrid[UV_GRID_COLS])
{
    float ui, vi;

    for (int i = 0; i < OUT_IMAGE_HEIGHT; ++i)
    {
        cv::Vec4b *row_ptr = outImage.ptr<cv::Vec4b>(i);
        for (int j = 0; j < OUT_IMAGE_WIDTH; ++j)
        {
            int uvGridIndex = i * OUT_IMAGE_WIDTH + j;
            ui = uGrid[uvGridIndex];
            vi = vGrid[uvGridIndex];

            if (ui < 5 || ui > 1920 - 5 || vi < 160 || vi > 1080)
            {
                // cout << "ui: " << ui << " vi: " << vi << endl;
            }
            else
            {
                cv::Vec3b *row_ptr_1 = inImage.ptr<cv::Vec3b>(static_cast<int>(vi));

                row_ptr[j][0] = static_cast<unsigned char>((row_ptr_1[static_cast<int>(ui)][0]));
                row_ptr[j][1] = static_cast<unsigned char>((row_ptr_1[static_cast<int>(ui)][1]));
                row_ptr[j][2] = static_cast<unsigned char>((row_ptr_1[static_cast<int>(ui)][2]));
            }
        }
    }
    // cout << outImage.at<Vec3b>(0, 100) << endl;

    return outImage;
}
int loadMappingArrays(int map_x[UV_GRID_COLS], int map_y[UV_GRID_COLS])
{
    std::ifstream infile_x("mapping_arr/ipm_undist_x.bin", std::ios::binary);
    std::ifstream infile_y("mapping_arr/ipm_undist_y.bin", std::ios::binary);

    if (!infile_x || !infile_y)
    {
        std::cout << "Cannot open file.\n";
        return 1;
    }
    for (int j = 0; j < UV_GRID_COLS; ++j)
    {
        infile_x.read((char *)&map_x[j], sizeof(int));
    }
    for (int j = 0; j < UV_GRID_COLS; ++j)
    {
        infile_y.read((char *)&map_y[j], sizeof(int));
    }
    infile_x.close();
    infile_y.close();
    return 0;
}
int loaduvGrid(int uGrid[UV_GRID_COLS], int vGrid[UV_GRID_COLS])
{
    std::ifstream infile("files/uv_grid.bin", std::ios::binary);

    if (!infile)
    {
        std::cout << "Cannot open file.\n";
        return 1;
    }
    float temp;

    for (int j = 0; j < UV_GRID_COLS; ++j)
    {
        infile.read((char *)&temp, sizeof(float));
        uGrid[j] = static_cast<int>(temp);
    }
    for (int j = 0; j < UV_GRID_COLS; ++j)
    {
        infile.read((char *)&temp, sizeof(float));
        vGrid[j] = static_cast<int>(temp);
    }
    infile.close();
    return 0;
}
int loaduvGridFromMatrix(int uGrid[UV_GRID_COLS], int vGrid[UV_GRID_COLS], const MatrixXd& uvGrd)
{
    for(int i = 0;i<UV_GRID_COLS;i++){
        uGrid[i] = static_cast<int>(uvGrd(0,i));
        vGrid[i] = static_cast<int>(uvGrd(1,i));
    }
    return 0;
}
Mat createEmptyAlphaMat(int rows, int cols)
{
    Mat mat(rows, cols, CV_8UC4, Scalar(0, 0, 0, 0));

    return mat;
}
void toUchar3(Mat frame, uchar3 *output, int width, int height)
{
    for (int i = 0; i < height; ++i)
    {
        cv::Vec3b *row_ptr = frame.ptr<cv::Vec3b>(i);
        for (int j = 0; j < width; ++j)
        {
            output[i * width + j].x = (unsigned char)row_ptr[j][0];
            output[i * width + j].y = (unsigned char)row_ptr[j][1];
            output[i * width + j].z = (unsigned char)row_ptr[j][2];
        }
    }
}
void toMat(uchar3 *input, Mat frame, int width, int height)
{
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            frame.at<Vec3b>(i, j)[0] = input[i * width + j].x;
            frame.at<Vec3b>(i, j)[1] = input[i * width + j].y;
            frame.at<Vec3b>(i, j)[2] = input[i * width + j].z;
        }
    }
}
void warpImage(uchar3 *inImage, uchar3 *outImage, int *uGrid, int *vGrid)
{
    int ui, vi;

    for (int i = 0; i < OUT_IMAGE_HEIGHT; ++i) // height
    {
        for (int j = 0; j < OUT_IMAGE_WIDTH; ++j) // width
        {
            int uvGridIndex = i * OUT_IMAGE_WIDTH + j;
            ui = uGrid[uvGridIndex];
            vi = vGrid[uvGridIndex];

            if (ui < 5 || ui > 1920 - 5 || vi < 160 || vi > 1080)
            {
            }
            else
            {
                outImage[i * OUT_IMAGE_WIDTH + j].x = inImage[vi * 1920 + ui].x;
                outImage[i * OUT_IMAGE_WIDTH + j].y = inImage[vi * 1920 + ui].y;
                outImage[i * OUT_IMAGE_WIDTH + j].z = inImage[vi * 1920 + ui].z;
            }
        }
    }
}
void undistort(uchar3 *inImage, uchar3 *outImage, int *map_x, int *map_y)
{
    int src_x, src_y;

    for (int i = 0; i < 1080; ++i)
    {
        for (int j = 0; j < 1920; ++j)
        {
            src_x = map_x[i * 1920 + j];
            src_y = map_y[i * 1920 + j];
            if (0 <= src_x && src_x < 1920 && 0 <= src_y && src_y < 1080)
            {
                outImage[i * 1920 + j].x = inImage[src_y * 1920 + src_x].x;
                outImage[i * 1920 + j].y = inImage[src_y * 1920 + src_x].y;
                outImage[i * 1920 + j].z = inImage[src_y * 1920 + src_x].z;
            }
        }
    }
}
int readArray(int *arr, const char *filename)
{
    std::ifstream infile(filename, std::ios::binary);

    if (!infile || arr == nullptr)
    {
        std::cout << "Cannot open file.\n";
        return 1;
    }
    float temp;

    for (int j = 0; j < 1920 * 1080; ++j)
    {
        infile.read((char *)&temp, sizeof(float));
        arr[j] = static_cast<int>(round(temp));
    }

    infile.close();
    return 0;
}
void equ(int *map_x, int *map_y, int *uGrid, int *vGrid)
{
    int *final_uGrid = new int[512 * 1024];
    int *final_vGrid = new int[512 * 1024];

    for (int i = 0; i < 512; ++i) // height of the final image
    {
        for (int j = 0; j < 1024; ++j) // width of the final image
        {
            // Calculate initial u, v coordinates for BEV
            int initial_ui = uGrid[i * 1024 + j];
            int initial_vi = vGrid[i * 1024 + j];
            if (initial_ui >= 0 && initial_ui <= 1920 && initial_vi >= 0 && initial_vi <= 1080)
            {
                // Then find the final u, v coordinates after undistortion
                final_uGrid[i * 1024 + j] = map_x[initial_vi * 1920 + initial_ui];
                final_vGrid[i * 1024 + j] = map_y[initial_vi * 1920 + initial_ui];
            }
            else
            {
                final_uGrid[i * 1024 + j] = -1;
                final_vGrid[i * 1024 + j] = -1;
            }
        }
    }
    delete final_uGrid;
    delete final_vGrid;
}
