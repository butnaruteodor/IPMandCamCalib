#include "ipm.h"
#include "config.h"

__global__ void rgbToIpm(uchar3 *input, uchar3 *output, int *uGrid, int *vGrid)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= OUT_IMAGE_WIDTH || y >= OUT_IMAGE_HEIGHT - 1)
        return;

    int uvIndex = y * OUT_IMAGE_WIDTH + x;

    int ui = uGrid[uvIndex];
    int vi = vGrid[uvIndex];

    if (ui >= 0 && ui < IN_IMAGE_WIDTH && vi >= 0 && vi < IN_IMAGE_HEIGHT)
    {
        int inIndex = vi * IN_IMAGE_WIDTH + ui;
        int outIndex = y * OUT_IMAGE_WIDTH + x;

        output[outIndex].x = input[inIndex].x;
        output[outIndex].y = input[inIndex].y;
        output[outIndex].z = input[inIndex].z;
    }
}
void warpImageK(uchar3 *input, uchar3 *output, int *uGrid, int *vGrid)
{
    dim3 blockDim(16, 16); // 16x16 threads per block
    dim3 gridDim((OUT_IMAGE_WIDTH + blockDim.x - 1) / blockDim.x, (OUT_IMAGE_HEIGHT + blockDim.y - 1) / blockDim.y);
    rgbToIpm<<<gridDim, blockDim>>>(input, output, uGrid, vGrid);
}
