#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include <cmath>
#include <numeric>
#include <vector>
#include <iostream>

#include <math.h>
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>

#include "util.h"
#include "frame.h"
#include "FEAmatcher.h"
#include "optimizer.h"
#include "cxxopts.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

#include <Eigen/Dense>

using namespace std;
using namespace Diasss;

#define PI 3.14159265


/************ Main Function  **************/

#ifndef DONT_USE_MAIN

int main(int argc, char** argv)
{
    int SUBFRAME_SIZE = 200; // the height size of subframe to crop;
    int KPS_TYPE = 2; // type of keypoint correspodences used: 0:ANNO, 1:SPARSE, 2:DENSE;
    int MIN_MATCHES = 20; // minimum number of matches between subframes to save an edge;

    std::string strImageFolder, strPoseFolder, strAltitudeFolder, strGroundRangeFolder, strAnnotationFolder, strPointCloudFolder;
    // --- read input data paths --- //
    {
      cxxopts::Options options("data_parsing", "Reads input files...");
      options.add_options()
          ("help", "Print help")
          ("image", "Input folder containing sss image files", cxxopts::value(strImageFolder))
          ("pose", "Input folder containing auv pose files", cxxopts::value(strPoseFolder))
          ("altitude", "Input folder containing auv altitude files", cxxopts::value(strAltitudeFolder))
          ("groundrange", "Input folder containing ground range files", cxxopts::value(strGroundRangeFolder))
          ("annotation", "Input folder containing annotation files", cxxopts::value(strAnnotationFolder))
          ("pointcloud", "Input folder containing point cloud files", cxxopts::value(strPointCloudFolder));

      auto result = options.parse(argc, argv);
      if (result.count("help")) {
        cout << options.help({"", "Group"}) << endl;
        exit(0);
      }
      if (result.count("image") == 0) {
        cout << "Please provide folder containing sss image files..." << endl;
        exit(0);
      }
      if (result.count("pose") == 0) {
        cout << "Please provide folder containing auv poses files..." << endl;
        exit(0);
      }
      if (result.count("altitude") == 0) {
        cout << "Please provide folder containing auv altitude files..." << endl;
        exit(0);
      }
      if (result.count("groundrange") == 0) {
        cout << "Please provide folder containing ground range files..." << endl;
        exit(0);
      }
      if (result.count("annotation") == 0) {
        cout << "Please provide folder containing annotation files..." << endl;
        exit(0);
      }
      if (result.count("pointcloud") == 0) {
        cout << "Please provide folder containing point cloud files..." << endl;
        exit(0);
      }
    }

    // --- parse input data --- //
    std::vector<cv::Mat> vmImgs;
    std::vector<cv::Mat> vmPoses;
    std::vector<std::vector<double>> vvAltts;
    std::vector<std::vector<double>> vvGranges;
    std::vector<cv::Mat> vmAnnos;
    std::vector<cv::Mat> vmPCs;
    Util::LoadInputData(strImageFolder,strPoseFolder,strAltitudeFolder,strGroundRangeFolder,strAnnotationFolder,strPointCloudFolder,
                        vmImgs,vmPoses,vvAltts,vvGranges,vmAnnos,vmPCs);

    // Util::AddNoiseToPose(vmPoses);
    // cout << "add noise to pose... " << endl;

    // --- construct frame --- //
    // int test_num = vmImgs.size();
    int test_num = 2;
    std::vector<Frame> test_frames;
    for (size_t i = 0; i < test_num; i++)
    {
        // cout << "constructing frame " << i << " ";
        test_frames.push_back(Frame(i,vmImgs[i],vmPoses[i],vvAltts[i],vvGranges[i],vmAnnos[i],vmPCs[i]));
        // cout << "complete!" << endl;

    }

    // --- find correspondences between each pair of frames --- //
    for (size_t i = 0; i < test_frames.size(); i++)
    {        
        for (size_t j = i+1; j < test_frames.size(); j++)
        {
            cout << "perform dense matching between " << i << " and " << j << " ... " << endl;
            // FEAmatcher::DenseMatchingS(test_frames[i],test_frames[j]);
            FEAmatcher::DenseMatchingD(test_frames[i],test_frames[j]);
            // Util::ShowAnnos(test_frames[i].img_id,test_frames[j].img_id,test_frames[i].norm_img,test_frames[j].norm_img,
            //     test_frames[i].anno_kps,test_frames[j].anno_kps);
            cout << "matching completed!" << endl;
        }
    }

    // --- divide a frame into multiple subframes--- //
    for (size_t i = 0; i < test_frames.size(); i++)
    {
        Util::FrameDividing(test_frames[i], SUBFRAME_SIZE, KPS_TYPE);    
    }

    // --- associate between subframes across different frames --- //
    for (size_t i = 0; i < test_frames.size(); i++)
    {
        for (size_t j = i+1; j < test_frames.size(); j++)
        {
            Util::SubFrameAssociating(test_frames[i], test_frames[j], MIN_MATCHES, KPS_TYPE); 
        }
           
    }

    // --- optimize trajectory between images --- //
    Optimizer::TrajOptimizationSubMap(test_frames);


    return 0;
}

#endif
