#include <Eigen/Dense>

using Eigen::MatrixXd;

struct CameraInfo {
    double yaw;
    double pitch;
    double focalLengthX;
    double focalLengthY;
    double opticalCenterX;
    double opticalCenterY;
};
MatrixXd GetVanishingPoint(const CameraInfo& cameraInfo);
void calcIpmMappingArrays();
