#include "util.h"

Mat warpImage(Mat inImage, Mat outImage, int uGrid[UV_GRID_COLS], int vGrid[UV_GRID_COLS])
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
int loaduvGridFromMatrix(int uGrid[UV_GRID_COLS], int vGrid[UV_GRID_COLS], const MatrixXd &uvGrd)
{
    for (int i = 0; i < UV_GRID_COLS; i++)
    {
        uGrid[i] = static_cast<int>(uvGrd(0, i));
        vGrid[i] = static_cast<int>(uvGrd(1, i));
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
int saveArray(int *arr, int size, const std::string &filename)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open())
    {
        std::cerr << "Couldn't open file for write: " << filename << std::endl;
        return 1;
    }
    ofs.write((const char *)(&size), sizeof(int));
    ofs.write((const char *)arr, size * sizeof(int));
    ofs.close();
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

    saveArray(final_uGrid, 512 * 1024, "mapping_arr/ipm_undist_x.bin");
    saveArray(final_vGrid, 512 * 1024, "mapping_arr/ipm_undist_y.bin");

    delete final_uGrid;
    delete final_vGrid;
}
VideoCapture getCamera(int camId)
{
#if PLATFORM == 1
    std::string gstreamer_pipeline(int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method)
    {
        return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
               std::to_string(capture_height) + ", framerate=(fraction)" + std::to_string(framerate) +
               "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
               std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true sync=false";
    }
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

#elif PLATFORM == 2
    VideoCapture cap(camId);
#endif
    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    if (width != IN_IMAGE_WIDTH || height != IN_IMAGE_HEIGHT)
    {
        std::cout << "Error: Camera resolution is not " << IN_IMAGE_WIDTH << "x" << IN_IMAGE_HEIGHT << std::endl;
        exit(1);
    }
    return cap;
}