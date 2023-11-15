#include "ipm.h"
#include "calc_arrays.h"
#include "../util/config.h"
#include "../util/util.h"

using namespace std;

int main()
{
    uchar3 *input, *output;
    int *uGrid, *vGrid;
    int *uGridCpu = new int[UV_GRID_COLS];
    int *vGridCpu = new int[UV_GRID_COLS];
    int *undistXCpu = new int[IN_IMAGE_HEIGHT * IN_IMAGE_WIDTH];
    int *undistYCpu = new int[IN_IMAGE_HEIGHT * IN_IMAGE_WIDTH];

    // Allocate Unified Memory – accessible from CPU or GPU for easier programming for jetson you would use zero copy memory with cudaAllocMapped
    cudaError_t err_x = cudaMallocManaged((void **)&input, IN_IMAGE_WIDTH * IN_IMAGE_HEIGHT * sizeof(uchar3));
    cudaError_t err_y = cudaMallocManaged((void **)&output, OUT_IMAGE_WIDTH * OUT_IMAGE_HEIGHT * sizeof(uchar3));

    // // Check for errors
    if (err_x != cudaSuccess || err_y != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate unified memory\n");
        return 1; // or handle error appropriately
    }

    int ret_x = readArray(undistXCpu, "undist_x.bin");
    int ret_y = readArray(undistYCpu, "undist_y.bin");
    if (ret_x == FAIL || ret_y == FAIL)
    {
        fprintf(stderr, "Failed to load undistort arrays. Do they exist?\nPerforming IPM without calibration.\n");
        return 1;
    }

    // In your program you would populate the cameraInfo and ipmInfo structs with your specific values
    CameraInfo cameraInfo;
    IpmInfo ipmInfo;

    cameraInfo.focalLengthX = (int)(IN_IMAGE_WIDTH / CAMERA_FOCAL_LENGTH) * CAMERA_SENSOR_WIDTH;
    cameraInfo.focalLengthY = (int)(IN_IMAGE_HEIGHT / CAMERA_FOCAL_LENGTH) * CAMERA_SENSOR_HEIGHT;
    cameraInfo.opticalCenterX = IN_IMAGE_WIDTH / 2;
    cameraInfo.opticalCenterY = IN_IMAGE_HEIGHT / 2;
    cameraInfo.cameraHeight = CAMERA_HEIGHT;
    cameraInfo.pitch = CAMERA_PITCH;
    cameraInfo.yaw = CAMERA_YAW;
    cameraInfo.roll = CAMERA_ROLL;

    ipmInfo.inputWidth = IN_IMAGE_WIDTH;
    ipmInfo.inputHeight = IN_IMAGE_HEIGHT;
    ipmInfo.left = IPM_LEFT;
    ipmInfo.right = IPM_RIGHT;
    ipmInfo.top = IPM_TOP;
    ipmInfo.bottom = IPM_BOTTOM;

    // Get the mapping arrays
    MatrixXd uvGrd = GetMappingArrays(cameraInfo, ipmInfo);
    // uvGrd is a 2xN matrix where N is OUT_IMAGE_WIDTH * OUT_IMAGE_HEIGHT, here we split it into two 1D arrays
    loaduvGridFromMatrix(uGridCpu, vGridCpu, uvGrd);

    // load the mapping arrays into the GPU
    cudaMalloc((void **)&uGrid, UV_GRID_COLS * sizeof(int));
    cudaMemcpy(uGrid, uGridCpu, UV_GRID_COLS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&vGrid, UV_GRID_COLS * sizeof(int));
    cudaMemcpy(vGrid, vGridCpu, UV_GRID_COLS * sizeof(int), cudaMemcpyHostToDevice);

    Mat frame;
    Mat outFrame(OUT_IMAGE_HEIGHT, OUT_IMAGE_WIDTH, CV_8UC3, Scalar(0, 0, 0));

    // Open the camera
    VideoCapture cam = getCamera(CAMERA_ID);
    if (!cam.isOpened())
    {
        std::cout << "Failed to open camera." << std::endl;
        return (-1);
    }
    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);

    std::cout << "Hit ESC to exit"
              << "\n"
              << "Hit s to save the final mapping arrays" << std::endl;
    while (true)
    {
        cv::Mat combined;
        if (!cam.read(frame))
        {
            std::cout << "Capture read error" << std::endl;
            break;
        }
        // Since i work with uchar3 i have to convert the frame to uchar3
        toUchar3(frame, input, frame.cols, frame.rows);
        // Warp the image
        warpImageK(input, output, uGrid, vGrid);
        // Wait for GPU to finish before doing something with the result
        cudaDeviceSynchronize();
        // Convert the output to Mat for visualization
        toMat(output, outFrame, OUT_IMAGE_WIDTH, OUT_IMAGE_HEIGHT);
        // Combine the original frame with the warped frame
        cv::resize(frame, frame, cv::Size(OUT_IMAGE_WIDTH, OUT_IMAGE_HEIGHT));
        cv::hconcat(frame, outFrame, combined);

        cv::imshow("Camera", combined);
        int keycode = cv::waitKey(1) & 0xff;
        if (keycode == 27)
        {
            break;
        }
        else if (keycode == 115)
        { // s key
            equ(undistXCpu, undistYCpu, uGridCpu, vGridCpu);
        }
    }

    cam.release();
    cv::destroyAllWindows();

    cudaFree(input);
    cudaFree(output);
    cudaFree(uGrid);
    cudaFree(vGrid);
    delete[] uGridCpu;
    delete[] vGridCpu;
    return 0;
}
