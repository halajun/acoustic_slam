
#include "util.h"
#include <random>

namespace Diasss
{

using namespace std;
using namespace cv;
using namespace gtsam;
using namespace Eigen;

#define PI 3.14159265358979323846264

bool SortPairInt(const pair<int,int> &a,
              const pair<int,int> &b)
{
    return (a.second > b.second);
}

bool closeEnough(const float& a, const float& b, const float& epsilon = std::numeric_limits<float>::epsilon()) 
{
    return (epsilon > std::abs(a - b));
}

float Util::ComputeIntersection(const std::vector<cv::Mat> &geo_img_s, const std::vector<cv::Mat> &geo_img_t)
{
    float output = 0.0;

    double sx_min, sy_min, sx_max, sy_max; // geometric border of source image
    double tx_min, tx_max, ty_min, ty_max; // geometric border of target image

    // get boundary of the source geo-image
    cv::minMaxLoc(geo_img_s[0], &sx_min, &sx_max);
    cv::minMaxLoc(geo_img_s[1], &sy_min, &sy_max);

    // get boundary of the target geo-image
    cv::minMaxLoc(geo_img_t[0], &tx_min, &tx_max);
    cv::minMaxLoc(geo_img_t[1], &ty_min, &ty_max);

    float x_dist_ol = std::min(sx_max, tx_max) - std::max(sx_min,tx_min);
    float y_dist_ol = std::min(ty_max, sy_max) - std::max(sy_min,ty_min);

    if (x_dist_ol>0 && y_dist_ol>0)
    {
        float area_ol = x_dist_ol*y_dist_ol;
        float area_s = std::abs(sx_max-sx_min)*std::abs(sy_max-sy_min);
        float area_t = std::abs(tx_max-tx_min)*std::abs(ty_max-ty_min);
        output = area_ol/(area_s+area_t-area_ol);
    }
    


    return output;

}

void Util::FrameDividing(Frame &CurFrame, const int &sf_height, const int &KPS_TYPE)
{
    float frame_w_half = CurFrame.norm_img.cols/2;

    // main loop (step size is sf_height)
    for (int i = sf_height; i < CurFrame.norm_img.rows; i=i+sf_height)
    {
        SubFrame sub_frame_tmp;

        // the ID of current sub-frame in the image
        sub_frame_tmp.subframe_id = (i-sf_height)/sf_height;

        // ping range of current sub-frame
        if (i+sf_height<=CurFrame.norm_img.rows)
        {
            sub_frame_tmp.end_ping = i;
            sub_frame_tmp.start_ping = i-sf_height;
            sub_frame_tmp.centre_ping = i-sf_height/2;
        }
        else
        {
            // in the case of getting to exceed the size of image,
            // includes the rest pings in the last frame;
            sub_frame_tmp.end_ping = CurFrame.norm_img.rows;
            sub_frame_tmp.start_ping = i-sf_height;
            sub_frame_tmp.centre_ping = (CurFrame.norm_img.rows+i-sf_height)/2;
        }

        cv::Mat kps_list;
        if (KPS_TYPE==0)
        {
            kps_list = CurFrame.anno_kps.clone();
            sub_frame_tmp.kps_type = 0;
        }
        else if(KPS_TYPE==1)
        {
            kps_list = CurFrame.corres_kps.clone();
            sub_frame_tmp.kps_type = 1;
        }
        else if(KPS_TYPE==2)
        {
            kps_list = CurFrame.corres_kps_dense.clone();
            sub_frame_tmp.kps_type = 2;
        }

        
        // loop on the keypoint list
        for (size_t i = 0; i < kps_list.rows; i++)
        {
            if (kps_list.at<int>(i,2)<sub_frame_tmp.end_ping && kps_list.at<int>(i,2)>=sub_frame_tmp.start_ping)
            {
                sub_frame_tmp.corres_ids.push_back(i);
            }           
        }
        
        // cout << "number of matches: " << sub_frame_tmp.corres_ids.size() << " " << sub_frame_tmp.start_ping << " " << sub_frame_tmp.end_ping << endl;

        CurFrame.subframes.push_back(sub_frame_tmp);




    }
    


    return;
}

void Util::SubFrameAssociating(Frame &SourceFrame, Frame &TargetFrame, const int &MIN_MATCHES, const int &KPS_TYPE)
{

    // get the list of keypoint correspodences in source frame
    cv::Mat kps_list_s;
    if (KPS_TYPE==0)
        kps_list_s = SourceFrame.anno_kps.clone();
    else if(KPS_TYPE==1)
        kps_list_s = SourceFrame.corres_kps.clone();
    else if(KPS_TYPE==2)
        kps_list_s = SourceFrame.corres_kps_dense.clone();

    // main loop
    for (size_t i = 0; i < SourceFrame.subframes.size(); i++)
    {

        // --- for matching displaying --- //
        std::vector<cv::KeyPoint> PreKeys, CurKeys;
        std::vector<cv::DMatch> TemperalMatches;
        int count = 0;

        // record associated subframe ids
        std::vector<int> asso_sf_ids(SourceFrame.subframes[i].corres_ids.size(),-1);

        for (size_t j = 0; j < SourceFrame.subframes[i].corres_ids.size(); j++)
        {
            int cur_id = SourceFrame.subframes[i].corres_ids[j];

            // only find corres in target frame and skip others
            if (kps_list_s.at<int>(cur_id,1)!=TargetFrame.img_id)
                continue;

            int kp_row_t = kps_list_s.at<int>(cur_id,4);

            // find which subframe in Targetframe, the current keypoint is corresponding to 
            for (size_t k = 0; k < TargetFrame.subframes.size(); k++)
            {
                if (kp_row_t<TargetFrame.subframes[k].end_ping && kp_row_t>=TargetFrame.subframes[k].start_ping)
                {
                    asso_sf_ids[j] = TargetFrame.subframes[k].subframe_id;
                    break;
                }                
            }
                        
        }

        // --- count and sort --- //
        // (1) count duplicates
        std::map<int, int> dups;
        for(int k : asso_sf_ids)
            ++dups[k];
        // (2) and sort them by descending order
        std::vector<std::pair<int, int> > sorted;
        for (auto k : dups)
            sorted.push_back(std::make_pair(k.first,k.second));
        std::sort(sorted.begin(), sorted.end(), SortPairInt);
        // (3) save if it meets minimum matches requirement
        for (size_t j = 0; j < sorted.size(); j++)
        {
            if (sorted[j].first!=-1 && sorted[j].second>MIN_MATCHES)
            {
                // --- save associated parent frame and subframe ids --- //
                SourceFrame.subframes[i].asso_sf_ids.push_back(std::make_pair(TargetFrame.img_id,sorted[j].first));
                // cout << "associated subframe: " << SourceFrame.img_id << "-" << SourceFrame.subframes[i].subframe_id << " <-> " << TargetFrame.img_id  << "-" <<  sorted[j].first << endl;
                // --- save the associated subframe correspondences' ids --- //
                std::vector<int> list_corres_ids;
                for (size_t k = 0; k < asso_sf_ids.size(); k++)
                {
                    if (asso_sf_ids[k]==sorted[j].first)
                    {
                        list_corres_ids.push_back(SourceFrame.subframes[i].corres_ids[k]);
                        
                        // for matching demonstration
                        PreKeys.push_back(cv::KeyPoint(kps_list_s.at<int>(SourceFrame.subframes[i].corres_ids[k],3),kps_list_s.at<int>(SourceFrame.subframes[i].corres_ids[k],2),0,0,0,-1));
                        CurKeys.push_back(cv::KeyPoint(kps_list_s.at<int>(SourceFrame.subframes[i].corres_ids[k],5),kps_list_s.at<int>(SourceFrame.subframes[i].corres_ids[k],4),0,0,0,-1));
                        TemperalMatches.push_back(cv::DMatch(count,count,0));
                        count = count + 1;
                    }
                    
                }
                SourceFrame.subframes[i].asso_sf_corres_ids.push_back(list_corres_ids);
                
            }
            
        }

    if (0)
    {
        // --- demonstrate --- //
        cv::Mat img_matches;
        cv::drawMatches(SourceFrame.norm_img, PreKeys, TargetFrame.norm_img, CurKeys, TemperalMatches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                        vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::namedWindow("temperal matches", cv::WINDOW_NORMAL);
        cv::imshow("temperal matches", img_matches);
        cv::waitKey(0);
    }
               

        

    }
    


    return;
}

void Util::AddNoiseToPose(std::vector<cv::Mat> &AllPose)
{
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,1.0);
    double noise_yaw = 1e-1/10*10; // 0.0005 in rad = 0.0286 in deg
    int counting = 0, N = 5000;
    cout << "noise_yaw to be: " << noise_yaw << endl;
    Matrix4d pose_dr_pre_tmp = Get4x4MatrixfromEulerandPosition(AllPose[0].at<double>(0,0),AllPose[0].at<double>(0,1),AllPose[0].at<double>(0,2), 
                                                                AllPose[0].at<double>(0,3),AllPose[0].at<double>(0,4),AllPose[0].at<double>(0,5));

    for (size_t i = 0; i < AllPose.size(); i++)
    {
        for (size_t j = 0; j < AllPose[i].rows; j++)
        {
            counting++;

            if (i==0 && j==0)
                continue;                             

            Matrix4d pose_dr_cur = Get4x4MatrixfromEulerandPosition(
                AllPose[i].at<double>(j,0),AllPose[i].at<double>(j,1),AllPose[i].at<double>(j,2), 
                AllPose[i].at<double>(j,3),AllPose[i].at<double>(j,4),AllPose[i].at<double>(j,5));

            Matrix4d pose_dr_pre;
            // if it's the first pose BUT NOT the first image, get previous pose from last image
            if (i!=0 && j==0)
            {
                int id = AllPose[i-1].rows - 1;
                pose_dr_pre = Get4x4MatrixfromEulerandPosition(
                    AllPose[i-1].at<double>(id,0),AllPose[i-1].at<double>(id,1),AllPose[i-1].at<double>(id,2), 
                    AllPose[i-1].at<double>(id,3),AllPose[i-1].at<double>(id,4),AllPose[i-1].at<double>(id,5));
                
            }
            // otherwise, get previous pose from last ping
            else
            {
                pose_dr_pre = Get4x4MatrixfromEulerandPosition(
                    AllPose[i].at<double>(j-1,0),AllPose[i].at<double>(j-1,1),AllPose[i].at<double>(j-1,2), 
                    AllPose[i].at<double>(j-1,3),AllPose[i].at<double>(j-1,4),AllPose[i].at<double>(j-1,5));
            }

            double new_yaw_noise = distribution(generator)*noise_yaw;
            Matrix4d add_noise = Get4x4MatrixfromEulerandPosition(0.0,0.0,new_yaw_noise,0.0, 0.0, 0.0);
            // if (i==0 && j<N)
            //     cout << "add_noise: " << endl << add_noise << endl;
            // // only add noise per 4 pings (sensor scan rate: 4 ping/sec)
            // if (counting%5==0||true) // 5 (1e-4), 10 (5e-5), 15 (2.5e-5), 20 (0.75e-4), 25 (0.375e-4),...
            // {
            //     cout << "Time to add noise at ping: " << counting << " with noise of " << new_yaw_noise << endl;
            //     Pose3 odo = pose_dr_pre.transformPoseTo(pose_dr_cur);
            //     nosiy_odo = odo.transformPoseFrom(add_noise); 
            // }
            // pose_dr_pre_noisy = pose_dr_pre; 
            Matrix4d odo = GetInverseMatrix4d(pose_dr_pre_tmp)*pose_dr_cur;
            // if (i==0 && j<N)            
            //         cout << "pose_dr_pre_tmp: " << endl << pose_dr_pre_tmp.matrix() << endl;
            // if (i==0 && j<N)            
            //         cout << "pose_dr_pre: " << endl << pose_dr_pre.matrix() << endl;
            Matrix4d pose_dr_cur_tmp = pose_dr_pre*odo;
            // if (i==0 && j<N)
            //     cout << "pose_dr_cur: " << endl << pose_dr_cur << endl;
            Matrix4d pose_dr_cur_noisy = pose_dr_cur_tmp*add_noise; 
            // // Matrix4d pose_dr_cur_noisy = pose_dr_cur;  
            // if (i==0 && j<N)
            //     cout << "pose_dr_cur_noisy: " << endl << pose_dr_cur_noisy << endl;


            AllPose[i].at<double>(j,2) = AllPose[i].at<double>(j,2) + new_yaw_noise;
            AllPose[i].at<double>(j,3) = pose_dr_cur_noisy(0,3);
            AllPose[i].at<double>(j,4) = pose_dr_cur_noisy(1,3);
            AllPose[i].at<double>(j,5) = pose_dr_cur_noisy(2,3);

            // save pose_dr_cur to pre_tmp
            pose_dr_pre_tmp = pose_dr_cur;


        }
    }

    // --- Save noisy pose results --- //
    ofstream save_result_1;
    string path1 = "../dr_poses_noisy_all.txt";
    save_result_1.open(path1.c_str(),ios::trunc);

    for (size_t i = 0; i < AllPose.size(); i++)
    {
        for (size_t j = 0; j < AllPose[i].rows; j++)
        {
            Pose3 save_pose = Pose3(
                    Rot3::Rodrigues(AllPose[i].at<double>(j,0),AllPose[i].at<double>(j,1),AllPose[i].at<double>(j,2)), 
                    Point3(AllPose[i].at<double>(j,3), AllPose[i].at<double>(j,4), AllPose[i].at<double>(j,5)));
            save_result_1 << fixed << setprecision(9) << save_pose.rotation().rpy()(0) << " " << save_pose.rotation().rpy()(1) << " "
                        << save_pose.rotation().rpy()(2) << " " << save_pose.x() << " " << save_pose.y() << " " << save_pose.z() << endl;

        }
    }

    save_result_1.close();
    // --------- end with saving ----------- //

    return;
}

gtsam::Quaternion Util::GetQuaternionfromEuler(const double &roll, const double &pitch, const double &yaw)
{

    // """
    // Convert an Euler angle to a quaternion.
    
    // Input
    //     :param roll: The roll (rotation around x-axis) angle in radians.
    //     :param pitch: The pitch (rotation around y-axis) angle in radians.
    //     :param yaw: The yaw (rotation around z-axis) angle in radians.
    
    // Output
    //     :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    // """

    double qx = sin(roll/2) * cos(pitch/2) * cos(yaw/2) - cos(roll/2) * sin(pitch/2) * sin(yaw/2);
    double qy = cos(roll/2) * sin(pitch/2) * cos(yaw/2) + sin(roll/2) * cos(pitch/2) * sin(yaw/2);
    double qz = cos(roll/2) * cos(pitch/2) * sin(yaw/2) - sin(roll/2) * sin(pitch/2) * cos(yaw/2);
    double qw = cos(roll/2) * cos(pitch/2) * cos(yaw/2) + sin(roll/2) * sin(pitch/2) * sin(yaw/2);
    
    return gtsam::Quaternion(qw, qx, qy, qz);

}

Eigen::Matrix4d Util::Get4x4MatrixfromEulerandPosition(const double &roll, const double &pitch, const double &yaw,
                                                       const double &x, const double &y, const double &z)
{

    Eigen::Matrix3d R;

    R = AngleAxisd(roll*PI, Vector3d::UnitX())
    * AngleAxisd(pitch*PI, Vector3d::UnitY())
    * AngleAxisd(yaw*PI, Vector3d::UnitZ());

    Eigen::Matrix4d output;

    output <<   R(0,0), R(0,1), R(0,2), x,
                R(1,0), R(1,1), R(1,2), y,
                R(2,0), R(2,1), R(2,2), z,
                0.0, 0.0, 0.0, 1.0;

    return output;
}

Eigen::Matrix4d Util::GetInverseMatrix4d(const Eigen::Matrix4d &input)
{


    Eigen::Matrix3d R = input.block<3, 3>(0, 0);
    Eigen::Matrix3d R_rotate = R.transpose();
    Eigen::Vector3d trans;
    trans << input(0, 3), input(1, 3), input(2, 3);
    Eigen::Vector3d trans_inv = -R_rotate*trans;

    Eigen::Matrix4d output;

    output <<   R_rotate(0,0), R_rotate(0,1), R_rotate(0,2), trans_inv(0),
                R_rotate(1,0), R_rotate(1,1), R_rotate(1,2), trans_inv(1),
                R_rotate(2,0), R_rotate(2,1), R_rotate(2,2), trans_inv(2),
                0.0, 0.0, 0.0, 1.0;

    return output;
}

Eigen::Vector3d Util::GetEulerAnglesfromRotation(const Eigen::Matrix3d R)
{
    double roll, pitch, yaw;

    // Assuming the angles are in radians.
    if (R(1,0) > 0.99998) { // singularity at north pole
        roll = 0;
        pitch = PI/2;
        yaw = atan2(R(0,2),R(2,2));
    }
    else if (R(1,0) < -0.99998) { // singularity at south pole
        roll = 0;
        pitch = -PI/2;
        yaw = atan2(R(0,2),R(2,2));
    }
    else
    {
        roll = atan2(-R(1,2),R(1,1));
        pitch = asin(R(1,0));
        yaw = atan2(-R(2,0),R(0,0));
    }

    Eigen::Vector3d euler;
    euler << roll, pitch, yaw;
    return euler;
 
}



void Util::LoadInputData(const std::string &strImageFolder, const std::string &strPoseFolder, const std::string &strAltitudeFolder, 
                         const std::string &strGroundRangeFolder, const std::string &strAnnotationFolder, const std::string &strPointCloudFolder,
                         std::vector<cv::Mat> &vmImgs, std::vector<cv::Mat> &vmPoses, std::vector<std::vector<double>> &vvAltts,
                         std::vector<std::vector<double>> &vvGranges, std::vector<cv::Mat> &vmAnnos, std::vector<cv::Mat> &vmPCs)
{
    // get data paths, ordered by name
    boost::filesystem::path path_img(strImageFolder);
    std::vector<boost::filesystem::path> path_img_ordered(
          boost::filesystem::directory_iterator(path_img)
          , boost::filesystem::directory_iterator{}
        );
    std::sort(path_img_ordered.begin(), path_img_ordered.end());

    boost::filesystem::path path_pose(strPoseFolder);
    std::vector<boost::filesystem::path> path_pose_ordered(
          boost::filesystem::directory_iterator(path_pose)
          , boost::filesystem::directory_iterator{}
        );
    std::sort(path_pose_ordered.begin(), path_pose_ordered.end());

    boost::filesystem::path path_altitude(strAltitudeFolder);
    std::vector<boost::filesystem::path> path_altitude_ordered(
          boost::filesystem::directory_iterator(path_altitude)
          , boost::filesystem::directory_iterator{}
        );
    std::sort(path_altitude_ordered.begin(), path_altitude_ordered.end());

    boost::filesystem::path path_groundrange(strGroundRangeFolder);
    std::vector<boost::filesystem::path> path_groundrange_ordered(
          boost::filesystem::directory_iterator(path_groundrange)
          , boost::filesystem::directory_iterator{}
        );
    std::sort(path_groundrange_ordered.begin(), path_groundrange_ordered.end());

    boost::filesystem::path path_annotation(strAnnotationFolder);
    std::vector<boost::filesystem::path> path_annotation_ordered(
          boost::filesystem::directory_iterator(path_annotation)
          , boost::filesystem::directory_iterator{}
        );
    std::sort(path_annotation_ordered.begin(), path_annotation_ordered.end());

    boost::filesystem::path path_pointcloud(strPointCloudFolder);
    std::vector<boost::filesystem::path> path_pointcloud_ordered(
          boost::filesystem::directory_iterator(path_pointcloud)
          , boost::filesystem::directory_iterator{}
        );
    std::sort(path_pointcloud_ordered.begin(), path_pointcloud_ordered.end());


    // get input sss images
    for(auto const& path : path_img_ordered)
    {
    //   std::cout << path << '\n';

      cv::Mat img_tmp;
      cv::FileStorage fs(path.string(),cv::FileStorage::READ);
      fs["ct_img"] >> img_tmp;
      fs.release();
      vmImgs.push_back(img_tmp);
      cout << "image size: "<< img_tmp.rows << " " << img_tmp.cols << endl;
      // cout.precision(10);
      // for (int i = 0; i < 100; i++)
      // {
      //     for (int j = 0; j < 100; j++)
      //     {
      //       cout << img_tmp.at<double>(i,j) << " ";
      //     }
      //     cout << endl;
      // }

    }

    // get input Dead-reckoning poses
    for(auto const& path : path_pose_ordered)
    {
    //   std::cout << path << '\n';

      cv::Mat pose_tmp;
      cv::FileStorage fs(path.string(),cv::FileStorage::READ);
      fs["auv_pose"] >> pose_tmp;
      fs.release();
      vmPoses.push_back(pose_tmp);
      cout << "pose size: " << pose_tmp.rows << " " << pose_tmp.cols << endl;
      // cout.precision(10);
      // for (int i = 0; i < 100; i++)
      // {
      //     for (int j = 0; j < 100; j++)
      //     {
      //       cout << pose_tmp.at<double>(i,j) << " ";
      //     }
      //     cout << endl;
      // }

    }

    // get auv altitude
    for(auto const& path : path_altitude_ordered)
    {
    //   std::cout << path << '\n';

      ifstream fAltt;
      fAltt.open(path.string().c_str());
      std::vector<double> vAltt;
      while(!fAltt.eof())
      {
          string s;
          getline(fAltt,s);
          if(!s.empty())
          {
              stringstream ss;
              ss << s;
              double altt_tmp;
              ss >> altt_tmp;
              vAltt.push_back(altt_tmp);
          }
      }
      fAltt.close();
      vvAltts.push_back(vAltt);
      cout << "alttitude size: " << vAltt.size() << endl;
      // for (double x : vAltt)
      //   cout << x << " ";
      // cout << endl;
    }

    // get ground range
    for(auto const& path : path_groundrange_ordered)
    {
    //   std::cout << path << '\n';

      ifstream fGRange;
      fGRange.open(path.string().c_str());
      std::vector<double> vGrange;
      while(!fGRange.eof())
      {
          string s;
          getline(fGRange,s);
          if(!s.empty())
          {
              stringstream ss;
              ss << s;
              double gr_tmp;
              ss >> gr_tmp;
              vGrange.push_back(gr_tmp);
          }
      }
      fGRange.close();
      vvGranges.push_back(vGrange);
      cout << "ground range size: " << vGrange.size() << endl;
      // for (double x : vGrange)
      //   cout << x << " ";
      // cout << endl;
    }

    // get input annotations of images
    for(auto const& path : path_annotation_ordered)
    {
    //   std::cout << path << '\n';

      cv::Mat anno_tmp;
      cv::FileStorage fs(path.string(),cv::FileStorage::READ);
      fs["anno_kps"] >> anno_tmp;
      fs.release();
      vmAnnos.push_back(anno_tmp);
      cout << "annotation size: "<< anno_tmp.rows << " " << anno_tmp.cols << endl;
      // cout.precision(10);
      // for (int i = 0; i < 100; i++)
      // {
      //     for (int j = 0; j < 100; j++)
      //     {
      //       cout << img_tmp.at<double>(i,j) << " ";
      //     }
      //     cout << endl;
      // }

    }

    // get input point cloud images
    for(auto const& path : path_pointcloud_ordered)
    {
    //   std::cout << path << '\n';

      cv::Mat pc_tmp;
      cv::FileStorage fs(path.string(),cv::FileStorage::READ);
      fs["reg_pc"] >> pc_tmp;
      fs.release();
      vmPCs.push_back(pc_tmp);
      cout << "image size: "<< pc_tmp.rows << " " << pc_tmp.cols << endl;
    //   cout.precision(10);
    //   for (int i = 0; i < pc_tmp.rows; i++)
    //   {
    //       for (int j = 0; j < pc_tmp.cols; j++)
    //       {
    //         if (pc_tmp.at<Vec3d>(i,j)[0]!=0 && pc_tmp.at<Vec3d>(i,j)[1]!=0 && pc_tmp.at<Vec3d>(i,j)[2]!=0)
    //         {
    //             cout << pc_tmp.at<Vec3d>(i,j)[0] << " " << pc_tmp.at<Vec3d>(i,j)[1] << " " << pc_tmp.at<Vec3d>(i,j)[2] << endl;
    //         }
    //       }
    //   }

    }

    return;                   
}

void Util::ShowAnnos(int &f1, int &f2, cv::Mat &img1, cv::Mat &img2, const cv::Mat &anno1, const cv::Mat &anno2)
{

    bool use_anno = 1;

    // --- load annotated keypoints --- //
    std::vector<cv::KeyPoint> PreKeys, CurKeys;
    std::vector<cv::DMatch> TemperalMatches;
    int count = 0;
    // --- from img1 to img2 ....... //
    for (size_t i = 0; i < anno1.rows; i++)
    {
        if (use_anno)
        {
            if (anno1.at<int>(i,1)==f2 && i%1==0)
            {
                PreKeys.push_back(cv::KeyPoint(anno1.at<int>(i,3),anno1.at<int>(i,2),0,0,0,-1));
                CurKeys.push_back(cv::KeyPoint(anno1.at<int>(i,5),anno1.at<int>(i,4),0,0,0,-1));
                TemperalMatches.push_back(cv::DMatch(count,count,0));
                count = count + 1;
                // cout << anno1.at<int>(i,2) << " " << anno1.at<int>(i,3) << " " << anno1.at<int>(i,4) << " " << anno1.at<int>(i,5) << endl;
            }  
        }
        else
        {
            if (anno1.at<double>(i,1)==f2)
            {
                PreKeys.push_back(cv::KeyPoint(anno1.at<double>(i,3),anno1.at<double>(i,2),0,0,0,-1));
                CurKeys.push_back(cv::KeyPoint(anno1.at<double>(i,5),anno1.at<double>(i,4),0,0,0,-1));
                TemperalMatches.push_back(cv::DMatch(count,count,0));
                count = count + 1;
                // cout << anno1.at<int>(i,2) << " " << anno1.at<int>(i,3) << " " << anno1.at<int>(i,4) << " " << anno1.at<int>(i,5) << endl;
            } 
        }
        
           
    }
    // // --- from img2 to img1 ....... //
    // for (size_t i = 0; i < anno2.rows; i++)
    // {
    //     if (anno2.at<int>(i,1)==f1)
    //     {
    //         PreKeys.push_back(cv::KeyPoint(anno2.at<int>(i,5),anno2.at<int>(i,4),0,0,0,-1));
    //         CurKeys.push_back(cv::KeyPoint(anno2.at<int>(i,3),anno2.at<int>(i,2),0,0,0,-1));
    //         TemperalMatches.push_back(cv::DMatch(count,count,0));
    //         count = count + 1;
    //     }     
    // }

    // cout << "number of matched keypoints: " << count << endl;

    // --- demonstrate --- //
    cv::Mat img_matches;
    cv::drawMatches(img1, PreKeys, img2, CurKeys, TemperalMatches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                    vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::namedWindow("temperal matches", cv::WINDOW_NORMAL);
    cv::imshow("temperal matches", img_matches);
    cv::waitKey(0);

    return;
}


} // namespace Diasss
