#include <Eigen/Dense>
#include <iostream>

using Eigen::MatrixXd;

struct CameraInfo {
    double yaw;
    double pitch;
    double roll;
    double focalLengthX;
    double focalLengthY;
    double opticalCenterX;
    double opticalCenterY;
    double cameraHeight;
};
struct IpmInfo{
    double inputWidth;
    double inputHeight;
    double left;
    double right;
    double top;
    double bottom;
};
MatrixXd GetVanishingPoint(const CameraInfo& cameraInfo);
MatrixXd TransformGround2Image(MatrixXd xyGrid, CameraInfo cameraInfo);
MatrixXd TransformImage2Ground(const MatrixXd& uvLimits, const CameraInfo& cameraInfo);
MatrixXd GetMappingArrays(const CameraInfo& cameraInfo, IpmInfo& ipmInfo);
