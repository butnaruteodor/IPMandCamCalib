#pragma once

#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>

__global__ void rgbToIpm(uchar3 *input, uchar3 *output, int *uGrid, int *vGrid);
extern "C" void warpImageK(uchar3 *input, uchar3 *output, int *uGrid, int *vGrid);
