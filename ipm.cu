#include "ipm.h"

#define UV_GRID_COLS 524288
#define OUT_IMAGE_WIDTH 1024
#define OUT_IMAGE_HEIGHT 512
// void warpImage(uchar3 *inImage, uchar3 *outImage)
// {
//     int ui, vi;

//     for (int i = 0; i < OUT_IMAGE_HEIGHT; ++i) // height
//     {
//         for (int j = 0; j < OUT_IMAGE_WIDTH; ++j) // width
//         {
//             int uvGridIndex = i * OUT_IMAGE_WIDTH + j;
//             ui = uvGrid[0][uvGridIndex];
//             vi = uvGrid[1][uvGridIndex];

//             if (ui < 5 || ui > 1920 - 5 || vi < 160 || vi > 1080)
//             {
//             }
//             else
//             {
//                 outImage[i][j][0] = inImage[vi][ui][0];
//                 outImage[i][j][1] = inImage[vi][ui][1];
//                 outImage[i][j][2] = inImage[vi][ui][2];
//             }
//         }
//     }
// }
// int loaduvGrid(int uGrid[UV_GRID_COLS], int vGrid[UV_GRID_COLS])
// {
//     std::ifstream infile("uv_grid.bin", std::ios::binary);

//     if (!infile)
//     {
//         std::cout << "Cannot open file.\n";
//         return 1;
//     }
//     float temp;

//     for (int j = 0; j < UV_GRID_COLS; ++j)
//     {
//         infile.read((char *)&temp, sizeof(float));
//         uGrid[j] = static_cast<int>(temp);
//     }
//     for (int j = 0; j < UV_GRID_COLS; ++j)
//     {
//         infile.read((char *)&temp, sizeof(float));
//         vGrid[j] = static_cast<int>(temp);
//     }
//     infile.close();
//     return 0;
// }
__global__ void rgbToIpm(uchar3 *input, uchar3 *output, int *uGrid, int *vGrid, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int uvIndex = y * width + x;

    int ui = uGrid[uvIndex];
    int vi = vGrid[uvIndex];

    if (ui >= 0 && ui < 1920 && vi >= 0 && vi < 1080)
    {
        int inIndex = vi * 1920 + ui;
        int outIndex = y * width + x;

        output[outIndex].x = input[inIndex].x;
        output[outIndex].y = input[inIndex].y;
        output[outIndex].z = input[inIndex].z;
    }
}
void warpImageK(uchar3 *input, uchar3 *output, int *uGrid, int *vGrid, int width, int height)
{
    dim3 blockDim(16, 16); // 16x16 threads per block
    dim3 gridDim((OUT_IMAGE_WIDTH + blockDim.x - 1) / blockDim.x, (OUT_IMAGE_HEIGHT + blockDim.y - 1) / blockDim.y);
    rgbToIpm<<<gridDim, blockDim>>>(input, output, uGrid, vGrid, 1024, 512);
}
// int main()
// {
//     uchar3 *input, *output;
//     int *uGrid, *vGrid;

//     cudaError_t err_x = cudaMallocManaged(&input, 1920 * 1080 * sizeof(uchar3));
//     cudaError_t err_y = cudaMallocManaged(&output, 1024 * 512 * sizeof(uchar3));
//     cudaError_t err_output = cudaMallocManaged(&uGrid, UV_GRID_COLS * sizeof(int));
//     cudaError_t err_output2 = cudaMallocManaged(&vGrid, UV_GRID_COLS * sizeof(int));
//     input[540 * 1920 + 960].x = 255;
//     input[540 * 1920 + 960].y = 255;
//     input[540 * 1920 + 960].z = 255;
//     // Check for errors
//     if (err_x != cudaSuccess || err_y != cudaSuccess || err_output != cudaSuccess || err_output2 != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to allocate unified memory\n");
//         return 1; // or handle error appropriately
//     }

//     loaduvGrid(uGrid, vGrid);

//     dim3 blockDim(16, 16); // 16x16 threads per block
//     dim3 gridDim((OUT_IMAGE_WIDTH + blockDim.x - 1) / blockDim.x, (OUT_IMAGE_HEIGHT + blockDim.y - 1) / blockDim.y);
//     rgbToIpm<<<gridDim, blockDim>>>(input, output, uGrid, vGrid, 1024, 512);
//     cudaDeviceSynchronize();

//     for (int i = 0; i < 1024 * 512; ++i)
//     {
//         if (output[i].x != 0)
//             std::cout << output[i].x << " " << output[i].y << " " << output[i].z << std::endl;
//     }

//     cudaFree(input);
//     cudaFree(output);
//     cudaFree(uGrid);
//     cudaFree(vGrid);
//     return 0;
// }
