
#ifndef UTIL_H
#define UTIL_H

#include<iostream>
#include <boost/filesystem.hpp>
#include <algorithm>
#include <limits>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Quaternion.h>
#include <gtsam/slam/BetweenFactor.h>

#include "subframe.h"
#include "frame.h"


namespace Diasss
{

    class Util
    {
    public:

        static gtsam::Quaternion GetQuaternionfromEuler(const double &roll, const double &pitch, const double &yaw);
        static Eigen::Matrix4d Get4x4MatrixfromEulerandPosition(const double &roll, const double &pitch, const double &yaw,
                                                            const double &x, const double &y, const double &z);
        static Eigen::Matrix4d GetInverseMatrix4d(const Eigen::Matrix4d &input);
        static Eigen::Vector3d GetEulerAnglesfromRotation(const Eigen::Matrix3d R);
        static void AddNoiseToPose(std::vector<cv::Mat> &AllPose);

        static float ComputeIntersection(const std::vector<cv::Mat> &geo_img_s, const std::vector<cv::Mat> &geo_img_t);

        static void FrameDividing(Frame &CurFrame, const int &sf_height, const int &KPS_TYPE);
        static void SubFrameAssociating(Frame &SourceFrame, Frame &TargetFrame, const int &MIN_MATCHES, const int &KPS_TYPE);

        static void LoadInputData(const std::string &strImageFolder, const std::string &strPoseFolder, const std::string &strAltitudeFolder, 
                                  const std::string &strGroundRangeFolder, const std::string &strAnnotationFolder, const std::string &strPointCloudFolder,
                                  std::vector<cv::Mat> &vmImgs, std::vector<cv::Mat> &vmPoses, std::vector<std::vector<double>> &vvAltts,
                                  std::vector<std::vector<double>> &vvGranges, std::vector<cv::Mat> &vmAnnos, std::vector<cv::Mat> &vmPCs);

        static void ShowAnnos(int &f1, int &f2, cv::Mat &img1, cv::Mat &img2, const cv::Mat &anno1, const cv::Mat &anno2);


    };

}

#endif