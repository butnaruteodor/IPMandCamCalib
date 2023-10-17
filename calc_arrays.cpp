#include "calc_arrays.h"
void calcIpmMappingArrays();

MatrixXd GetVanishingPoint(const CameraInfo &cameraInfo)
{
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

MatrixXd TransformGround2Image(MatrixXd xyGrid, CameraInfo cameraInfo)
{
       MatrixXd inPoints2 = xyGrid.block(0, 0, 2, xyGrid.cols()); // First two rows
       // Create a row vector with all values as -cameraInfo.cameraHeight
       Eigen::RowVectorXd inPointsr3 = Eigen::RowVectorXd::Constant(xyGrid.cols(), -cameraInfo.cameraHeight);
       // Stack inPoints2 and inPointsr3 vertically to create inPoints3
       MatrixXd inPoints3(3, xyGrid.cols());
       inPoints3 << inPoints2,
           inPointsr3;

       double c1 = std::cos(cameraInfo.pitch * M_PI / 180);
       double s1 = std::sin(cameraInfo.pitch * M_PI / 180);
       double c2 = std::cos(cameraInfo.yaw * M_PI / 180);
       double s2 = std::sin(cameraInfo.yaw * M_PI / 180);

       MatrixXd matp(3, 3);
       matp << cameraInfo.focalLengthX * c2 + c1 * s2 * cameraInfo.opticalCenterX,
           -cameraInfo.focalLengthX * s2 + c1 * c2 * cameraInfo.opticalCenterX,
           -s1 * cameraInfo.opticalCenterX,

           s2 * (-cameraInfo.focalLengthY * s1 + c1 * cameraInfo.opticalCenterY),
           c2 * (-cameraInfo.focalLengthY * s1 + c1 * cameraInfo.opticalCenterY),
           -cameraInfo.focalLengthY * c1 - s1 * cameraInfo.opticalCenterY,

           c1 * s2, c1 * c2, -s1;

       inPoints3 = matp * inPoints3;
       inPointsr3 = inPoints3.row(2);       // Third row
       Eigen::RowVectorXd div = inPointsr3; // Division row vector
       // Perform division operation for each row
       inPoints3.row(0) = inPoints3.row(0).array() / div.array();
       inPoints3.row(1) = inPoints3.row(1).array() / div.array();

       inPoints2 = inPoints3.block(0, 0, 2, inPoints3.cols()); // First two rows

       return inPoints2; // Return the result as uvGrid
}

MatrixXd TransformImage2Ground(const MatrixXd &uvLimits, const CameraInfo &cameraInfo)
{
       int row = uvLimits.rows();
       int col = uvLimits.cols();
       MatrixXd inPoints4 = MatrixXd::Zero(row + 2, col);
       inPoints4.block(0, 0, row, col) = uvLimits;
       inPoints4.row(2).setOnes();

       MatrixXd inPoints3 = inPoints4.block(0, 0, 3, col);

       double c1 = std::cos(cameraInfo.pitch * M_PI / 180);
       double s1 = std::sin(cameraInfo.pitch * M_PI / 180);
       double c2 = std::cos(cameraInfo.yaw * M_PI / 180);
       double s2 = std::sin(cameraInfo.yaw * M_PI / 180);

       MatrixXd matp(4, 3);
       matp << -cameraInfo.cameraHeight * c2 / cameraInfo.focalLengthX,
           cameraInfo.cameraHeight * s1 * s2 / cameraInfo.focalLengthY,
           (cameraInfo.cameraHeight * c2 * cameraInfo.opticalCenterX / cameraInfo.focalLengthX) -
               (cameraInfo.cameraHeight * s1 * s2 * cameraInfo.opticalCenterY / cameraInfo.focalLengthY) -
               cameraInfo.cameraHeight * c1 * s2,

           cameraInfo.cameraHeight * s2 / cameraInfo.focalLengthX,
           cameraInfo.cameraHeight * s1 * c2 / cameraInfo.focalLengthY,
           (-cameraInfo.cameraHeight * s2 * cameraInfo.opticalCenterX / cameraInfo.focalLengthX) -
               (cameraInfo.cameraHeight * s1 * c2 * cameraInfo.opticalCenterY / cameraInfo.focalLengthY) -
               cameraInfo.cameraHeight * c1 * c2,

           0,
           cameraInfo.cameraHeight * c1 / cameraInfo.focalLengthY,
           (-cameraInfo.cameraHeight * c1 * cameraInfo.opticalCenterY / cameraInfo.focalLengthY) +
               cameraInfo.cameraHeight * s1,

           0,
           -c1 / cameraInfo.focalLengthY,
           (c1 * cameraInfo.opticalCenterY / cameraInfo.focalLengthY) - s1;

       MatrixXd inPoints4Transformed = matp * inPoints3;
       MatrixXd div = inPoints4Transformed.row(3);

       for (int i = 0; i < 4; ++i)
       {
              inPoints4Transformed.row(i).array() /= div.array();
       }

       MatrixXd xyLimits = inPoints4Transformed.block(0, 0, 2, col);

       return xyLimits;
}
MatrixXd GetMappingArrays(const CameraInfo &cameraInfo, IpmInfo &ipmInfo)
{
       MatrixXd vp = GetVanishingPoint(cameraInfo);

       double vp_x = vp(0, 0);
       double vp_y = vp(1, 0);

       ipmInfo.top = (double)std::max(vp_y, ipmInfo.top);

       MatrixXd uvLimitsp(2, 4);
       uvLimitsp << vp_x, ipmInfo.right, ipmInfo.left, vp_x,
           ipmInfo.top, ipmInfo.top, ipmInfo.top, ipmInfo.bottom;

       MatrixXd xyLimits = TransformImage2Ground(uvLimitsp,cameraInfo);

       double xfMin = xyLimits.row(0).minCoeff();
       double xfMax = xyLimits.row(0).maxCoeff();
       double yfMin = xyLimits.row(1).minCoeff();
       double yfMax = xyLimits.row(1).maxCoeff();

       double xyRatio = (xfMax - xfMin)/(yfMax-yfMin);
       int outRow = 512;
       int outCol = 1024;

       double stepRow = (yfMax - yfMin) / outRow;
       double stepCol = (xfMax - xfMin) / outCol;

       MatrixXd xyGrid = MatrixXd::Zero(2, outRow*outCol);

       double y = yfMax - 0.5 * stepRow;

       for(int i = 0;i<outRow;i++){
              double x = xfMin + 0.5 * stepCol;
              for(int j = 0;j<outCol;j++){
                     xyGrid(0,i*outCol+j) = x;
                     xyGrid(1,i*outCol+j) = y;
                     x += stepCol;
              }
              y -= stepRow;
       }
       MatrixXd uvGrid = TransformGround2Image(xyGrid,cameraInfo);
       return uvGrid;
}