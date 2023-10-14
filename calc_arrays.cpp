#include "calc_arrays.h"
void calcIpmMappingArrays();

MatrixXd GetVanishingPoint(const CameraInfo& cameraInfo) {
    MatrixXd vpp(3, 1);
    vpp << std::sin(cameraInfo.yaw * M_PI / 180) / std::cos(cameraInfo.pitch * M_PI / 180),
           std::cos(cameraInfo.yaw * M_PI / 180) / std::cos(cameraInfo.pitch * M_PI / 180),
           0;

    MatrixXd tyawp(3, 3);
    tyawp << std::cos(cameraInfo.yaw * M_PI / 180), -std::sin(cameraInfo.yaw * M_PI / 180), 0,
             std::sin(cameraInfo.yaw * M_PI / 180), std::cos(cameraInfo.yaw * M_PI / 180), 0,
             0, 0, 1;

    MatrixXd tpitchp(3, 3);
    tpitchp << 1, 0, 0,
               0, -std::sin(cameraInfo.pitch * M_PI / 180), -std::cos(cameraInfo.pitch * M_PI / 180),
               0, std::cos(cameraInfo.pitch * M_PI / 180), -std::sin(cameraInfo.pitch * M_PI / 180);

    MatrixXd t1p(3, 3);
    t1p << cameraInfo.focalLengthX, 0, cameraInfo.opticalCenterX,
           0, cameraInfo.focalLengthY, cameraInfo.opticalCenterY,
           0, 0, 1;

    MatrixXd transform = t1p * (tyawp * tpitchp);
    MatrixXd vp = transform * vpp;
    return vp;
}
