#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>

#include "../util/config.h"
#include "../util/util.h"
using namespace std;

#define CALIB_PATH "calib/images/*.png"
#define MANUAL_CALIB_PATH "calib/frame445.png"

// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{6, 9};

int main()
{
  // Creating vector to store vectors of 3D points for each checkerboard image
  std::vector<std::vector<cv::Point3f>> objpoints;

  // Creating vector to store vectors of 2D points for each checkerboard image
  std::vector<std::vector<cv::Point2f>> imgpoints;

  // Defining the world coordinates for 3D points
  std::vector<cv::Point3f> objp;
  for (int i{0}; i < CHECKERBOARD[1]; i++)
  {
    for (int j{0}; j < CHECKERBOARD[0]; j++)
      objp.push_back(cv::Point3f(j, i, 0));
  }

  // Extracting path of individual image stored in a given directory
  std::vector<cv::String> images;
  // Path of the folder containing checkerboard images
  std::string path = CALIB_PATH;

  cv::glob(path, images);

  cv::Mat frame, gray;
  // vector to store the pixel coordinates of detected checker board corners
  std::vector<cv::Point2f> corner_pts;
  bool success;

  // Looping over all the images in the directory
  for (int i{0}; i < images.size(); i++)
  {
    frame = cv::imread(images[i]);
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // Finding checker board corners
    // If desired number of corners are found in the image then success = true
    success = cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

    /*
     * If desired number of corner are detected,
     * we refine the pixel coordinates and display
     * them on the images of checker board
     */
    if (success)
    {
      cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);

      // refining pixel coordinates for given 2d points.
      cv::cornerSubPix(gray, corner_pts, cv::Size(11, 11), cv::Size(-1, -1), criteria);

      // Displaying the detected corner points on the checker board
      cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);

      objpoints.push_back(objp);
      imgpoints.push_back(corner_pts);
    }

    cv::imshow("Image", frame);
    cv::waitKey(0);
  }

  cv::Mat cameraMatrix, distCoeffs, R, T;

  /*
   * Performing camera calibration by
   * passing the value of known 3D points (objpoints)
   * and corresponding pixel coordinates of the
   * detected corners (imgpoints)
   */
  cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, R, T);

  cv::Mat image = cv::imread(MANUAL_CALIB_PATH);
  cv::Mat controlImage = cv::Mat::zeros(200, 400, CV_8UC3);

  cv::Mat map_x, map_y;
  int *undistX = new int[IN_IMAGE_HEIGHT * IN_IMAGE_WIDTH];
  int *undistY = new int[IN_IMAGE_HEIGHT * IN_IMAGE_WIDTH];

  cv::namedWindow("Control Window", 1);
  cv::namedWindow("Main Window", 1);

  int rad1 = 5000;
  int rad2 = 5000;
  int rad3 = 5000;
  int tan1 = 5000;
  int tan2 = 5000;
  cv::createTrackbar("k1", "Control Window", &rad1, 10000);
  cv::createTrackbar("k2", "Control Window", &rad2, 10000);
  cv::createTrackbar("p1", "Control Window", &tan1, 10000);
  cv::createTrackbar("p2", "Control Window", &tan2, 10000);
  cv::createTrackbar("k3", "Control Window", &rad3, 10000);

  int fontFace = cv::FONT_HERSHEY_SIMPLEX;
  double fontScale = 0.5;
  cv::Scalar color(255, 255, 255); // White color
  int thickness = 1;
  int lineType = cv::LINE_8;

  while (true)
  {
    cv::Mat res = cv::Mat::zeros(720, 1280, CV_8UC3);
    cv::Mat dst = cv::Mat::zeros(1080, 1920, CV_8UC3);
    cv::Mat controlImage = cv::Mat::zeros(200, 400, CV_8UC3);

    // update distCoeffs
    distCoeffs.at<double>(0, 0) = (rad1 - 5000) / 5000.0;
    distCoeffs.at<double>(0, 1) = (rad2 - 5000) / 5000.0;
    distCoeffs.at<double>(0, 2) = (tan1 - 5000) / 5000.0;
    distCoeffs.at<double>(0, 3) = (tan2 - 5000) / 5000.0;
    distCoeffs.at<double>(0, 4) = (rad3 - 5000) / 5000.0;
    cv::initUndistortRectifyMap(
        cameraMatrix, distCoeffs, cv::Mat(), cameraMatrix, cv::Size(1920, 1080), CV_32FC1, map_x, map_y);

    // dst = custom_remap(image, map_x, map_y);
    cv::remap(image, dst, map_x, map_y, cv::INTER_LINEAR);
    getMapFromMat(undistX, map_x);
    getMapFromMat(undistY, map_y);
    cv::resize(dst, res, cv::Size(1280, 720), -1);

    cv::putText(controlImage, "k1 = " + to_string(distCoeffs.at<double>(0, 0)), cv::Point(10, 20), fontFace, fontScale, color, thickness, lineType, false);
    cv::putText(controlImage, "k2 = " + to_string(distCoeffs.at<double>(0, 1)), cv::Point(10, 40), fontFace, fontScale, color, thickness, lineType, false);
    cv::putText(controlImage, "p1 = " + to_string(distCoeffs.at<double>(0, 2)), cv::Point(10, 60), fontFace, fontScale, color, thickness, lineType, false);
    cv::putText(controlImage, "p2 = " + to_string(distCoeffs.at<double>(0, 3)), cv::Point(10, 80), fontFace, fontScale, color, thickness, lineType, false);
    cv::putText(controlImage, "k3 = " + to_string(distCoeffs.at<double>(0, 4)), cv::Point(10, 100), fontFace, fontScale, color, thickness, lineType, false);

    cv::imshow("Control Window", controlImage);
    cv::imshow("Main Window", dst);
    int iKey = cv::waitKey(50);
    if (iKey == 27)
    {
      break;
    }
    else if (iKey == 115)
    { // s key
      // save the mapping arrays map_x and map_y
      saveArray(undistX, IN_IMAGE_WIDTH * IN_IMAGE_HEIGHT, "undist_x.bin");
      saveArray(undistY, IN_IMAGE_WIDTH * IN_IMAGE_HEIGHT, "undist_y.bin");
    }
  }
  cv::destroyAllWindows();

  return 0;
}
