#pragma once
// Platform
#define PLATFORM 2 // The platform that the code is running on, Jetson(1) or x86(2)

// Inverse perspective mapping configs
#define UV_GRID_COLS 524288  // Number of elements in a mapping array ie. 1024*512
#define OUT_IMAGE_WIDTH 1024 // The width of the image that the ipm algorithm will produce
#define OUT_IMAGE_HEIGHT 512 // The height of the image that the ipm algorithm will produce

#define IN_IMAGE_WIDTH 1920  // The width of the input image
#define IN_IMAGE_HEIGHT 1080 // The height of the input image

#define CAMERA_HEIGHT 267 // The height of the camera in cm
#define CAMERA_PITCH 13.0 // The pitch of the camera in degrees
#define CAMERA_YAW 0      // The yaw of the camera in degrees
#define CAMERA_ROLL 0     // The roll of the camera in degrees

// Here you basically crop the image to the area that you are interested in
// Considering that a 1920x1080 image is used, the following values will crop the original image before the ipm algorithm is applied
// Here the image is cropped to 1910x545 before the ipm algorithm is applied
#define IPM_LEFT 5                       // Ignore 5 cols of pixels of the left side of the image
#define IPM_RIGHT IN_IMAGE_WIDTH - 5     // Ignore 5 cols of pixels of the right side of the image
#define IPM_TOP 375                      // Ignore 375 rows of pixels of the top of the image
#define IPM_BOTTOM IN_IMAGE_HEIGHT - 160 // Ignore 160 rows of pixels of the bottom of the image

// Camera
#define CAMERA_ID 0 // The id of the camera that will be used

// Camera calibration configs
