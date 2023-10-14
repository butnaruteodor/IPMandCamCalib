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
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "ipm.h"

using namespace std;
using namespace cv;

// struct uchar3
// {
//     unsigned char x, y, z;
// };

#define UV_GRID_ROWS 2
#define UV_GRID_COLS 524288

#define OUT_IMAGE_WIDTH 1024
#define OUT_IMAGE_HEIGHT 512
uchar3 *aux;
Mat outImage(OUT_IMAGE_HEIGHT, OUT_IMAGE_WIDTH, CV_8UC3, Scalar(0, 0, 0));
static unsigned char imgBuffer[OUT_IMAGE_HEIGHT][OUT_IMAGE_WIDTH][3] = {0};

int readArray(int *arr, const char *filename);
void undistort(uchar3 *inImage, uchar3 *outImage, int *map_x, int *map_y);
void warpImage(uchar3 *inImage, uchar3 *outImage, int *uGrid, int *vGrid);
int loaduvGrid(int uGrid[UV_GRID_COLS], int vGrid[UV_GRID_COLS]);
int loadMappingArrays(int map_x[UV_GRID_COLS],int map_y[UV_GRID_COLS]);
Mat warpImage(Mat inImage, int uGrid[UV_GRID_COLS], int vGrid[UV_GRID_COLS]);
Mat createEmptyAlphaMat(int rows, int cols);
void toUchar3(Mat frame, uchar3 *output, int width, int height);
void toMat(uchar3 *input, Mat frame, int width, int height);
void equ(uchar3 *inImage, uchar3 *outImage, int *map_x, int *map_y, int *uGrid, int *vGrid);
int main()
{
    uchar3 *input, *output, *output_calib;
    int *uGrid, *vGrid;
    int *uGridCpu = new int[UV_GRID_COLS];
    int *vGridCpu = new int[UV_GRID_COLS];

    cudaError_t err_x = cudaMallocManaged((void**)&input, 1920 * 1080 * sizeof(uchar3));
    cudaError_t err_y = cudaMallocManaged((void**)&output, 1024 * 512 * sizeof(uchar3));
    cudaError_t err_y1 = cudaMallocManaged((void**)&output_calib, 1920 * 1080 * sizeof(uchar3));
    cudaError_t err_y2 = cudaMallocManaged((void**)&aux, 1920 * 1080 * sizeof(uchar3));
    // cudaError_t err_output = cudaMallocManaged(&uGrid, UV_GRID_COLS * sizeof(int));
    // cudaError_t err_output2 = cudaMallocManaged(&vGrid, UV_GRID_COLS * sizeof(int));

    // // Check for errors
    if (err_x != cudaSuccess || err_y != cudaSuccess /*|| err_output != cudaSuccess || err_output2 != cudaSuccess*/)
    {
        fprintf(stderr, "Failed to allocate unified memory\n");
        return 1; // or handle error appropriately
    }

    Mat frame;
    Mat outFrame;
    //--- INITIALIZE VIDEOCAPTURE
    VideoCapture cap;
    // Load uv grid from file

    //loaduvGrid(uGridCpu, vGridCpu);
    loadMappingArrays(uGridCpu,vGridCpu);

    cudaMalloc((void **)&uGrid, UV_GRID_COLS * sizeof(int));
    cudaMemcpy(uGrid, uGridCpu, UV_GRID_COLS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&vGrid, UV_GRID_COLS * sizeof(int));
    cudaMemcpy(vGrid, vGridCpu, UV_GRID_COLS * sizeof(int), cudaMemcpyHostToDevice);

    /**************Delete*********/

    // for (int i = 428; i <= 2617; i++)
    // {
    //     std::string filename = "imgs/frame" + std::to_string(i) + ".png"; // Assuming PNG format
    //     frame = cv::imread(filename);

    //     if (frame.empty())
    //     {
    //         // std::cout << "Could not read image: " << filename << std::endl;
    //         continue; // Skip this iteration if the image can't be read
    //     }
    //     toUchar3(frame, input, frame.cols, frame.rows);
    //     warpImageK(input, output, uGrid, vGrid, frame.cols, frame.rows);
    //     cudaDeviceSynchronize();
    //     toMat(output, outImage, OUT_IMAGE_WIDTH, OUT_IMAGE_HEIGHT);

    //     cv::imshow("Image", outImage);
    //     cv::waitKey(1); // Wait for 1 ms, change this value as needed
    // }

    /**********************************/

    Mat img = imread("images/calib.jpg", IMREAD_COLOR);
    if (img.empty())
    {
        std::cout << "Could not read the image: "
                  << "path" << std::endl;
        return 1;
    }
    toUchar3(img, input, img.cols, img.rows);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //warpImageK(input, output, uGrid, vGrid, img.cols, img.rows);
    //warpImageK(input, output, uGrid, vGrid, img.cols, img.rows);
    //while(true){
    // Record the start event
    cudaEventRecord(start, 0);

    warpImageK(input, output, uGrid, vGrid, img.cols, img.rows);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Elapsed time: %f ms\n", milliseconds);
   // }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /***********Delete*/
    // int *map_x = new int[1920 * 1080];
    // int *map_y = new int[1920 * 1080];
    // readArray(map_x, "files/map_x.bin");
    // readArray(map_y, "files/map_y.bin");
    // Mat outImageCalib(512, 1024, CV_8UC3, Scalar(0, 0, 0));
    // //undistort(input, output_calib, map_x, map_y);
    // warpImage(input, output, uGridCpu, vGridCpu);
    // //equ(input, output, map_x, map_y, uGridCpu, vGridCpu);
    // //cout << "After equ\n";
    // toMat(output, outImageCalib, 1024, 512);
    /****************/
    cudaDeviceSynchronize();
    toMat(output, outImage, OUT_IMAGE_WIDTH, OUT_IMAGE_HEIGHT);
    imwrite("delete.png",outImage);
    imshow("Display window", outImage);
    int k = waitKey(0); // Wait for a keystroke in the window

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
int loadMappingArrays(int map_x[UV_GRID_COLS],int map_y[UV_GRID_COLS]){
    std::ifstream infile_x("files/ipm_undist_x.bin", std::ios::binary);
    std::ifstream infile_y("files/ipm_undist_y.bin", std::ios::binary);

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
void equ(uchar3 *inImage, uchar3 *outImage, int *map_x, int *map_y, int *uGrid, int *vGrid)
{
    // int ui, vi;
    // for (int i = 0; i < 1080; ++i)
    // {
    //     for (int j = 0; j < 1920; ++j)
    //     {
    //         ui = map_x[i * 1920 + j];
    //         vi = map_y[i * 1920 + j];
    //         if (0 <= ui && ui < 1920 && 0 <= vi && vi < 1080)
    //         {
    //             aux[i * 1920 + j].x = inImage[vi * 1920 + ui].x;
    //             aux[i * 1920 + j].y = inImage[vi * 1920 + ui].y;
    //             aux[i * 1920 + j].z = inImage[vi * 1920 + ui].z;
    //         }
    //     }
    // }

    // for (int i = 0; i < 512; ++i) // height
    // {
    //     for (int j = 0; j < 1024; ++j) // width
    //     {
    //         ui = uGrid[i * 1024 + j];
    //         vi = vGrid[i * 1024 + j];

    //         if (ui >= 0 && ui <= 1920 && vi >= 0 && vi <= 1080)
    //         {
    //             outImage[i * 1024 + j].x = aux[vi * 1920 + ui].x;
    //             outImage[i * 1024 + j].y = aux[vi * 1920 + ui].y;
    //             outImage[i * 1024 + j].z = aux[vi * 1920 + ui].z;
    //         }
    //     }
    // }

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
    // Use the final mapping to transform image
    for (int i = 0; i < 512; ++i) // height
    {
        for (int j = 0; j < 1024; ++j) // width
        {
            int ui = final_uGrid[i * 1024 + j];
            int vi = final_vGrid[i * 1024 + j];

            if (ui >= 0 && ui < 1920 && vi >= 0 && vi < 1080)
            {
                // cout << "ui: " << ui << " vi: " << vi << "\n";
                outImage[i * 1024 + j].x = inImage[vi * 1920 + ui].x;
                outImage[i * 1024 + j].y = inImage[vi * 1920 + ui].y;
                outImage[i * 1024 + j].z = inImage[vi * 1920 + ui].z;
            }
        }
    }
    delete final_uGrid;
    delete final_vGrid;
}
