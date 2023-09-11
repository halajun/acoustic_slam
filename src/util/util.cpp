
#include "util.h"
#include <random>

namespace Diasss
{

using namespace std;
using namespace cv;
using namespace gtsam;

#define PI 3.14159265359

bool SortPairInt(const pair<int,int> &a,
              const pair<int,int> &b)
{
    return (a.second > b.second);
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
    float noise_yaw = 0.00005; // 0.0005 in rad = 0.0286 in deg
    Pose3 add_noise(Rot3::Rodrigues(0, 0, distribution(generator)*noise_yaw),Point3(0, 0, 0));
    Pose3 pose_dr_pre_tmp = Pose3(Rot3::Rodrigues(AllPose[0].at<double>(0,0),AllPose[0].at<double>(0,1),AllPose[0].at<double>(0,2)), 
                    Point3(AllPose[0].at<double>(0,3), AllPose[0].at<double>(0,4), AllPose[0].at<double>(0,5)));

    for (size_t i = 0; i < AllPose.size(); i++)
    {
        for (size_t j = 0; j < AllPose[i].rows; j++)
        {
            if (i==0 && j==0)
                continue;                             

            Pose3 pose_dr_cur = Pose3(
                Rot3::Rodrigues(AllPose[i].at<double>(j,0),AllPose[i].at<double>(j,1),AllPose[i].at<double>(j,2)), 
                Point3(AllPose[i].at<double>(j,3), AllPose[i].at<double>(j,4), AllPose[i].at<double>(j,5)));

            Pose3 pose_dr_pre;
            // if it's the first pose BUT NOT the first image, get previous pose from last image
            if (i!=0 && j==0)
            {
                int id = AllPose[i-1].rows - 1;
                pose_dr_pre = Pose3(
                    Rot3::Rodrigues(AllPose[i-1].at<double>(id,0),AllPose[i-1].at<double>(id,1),AllPose[i-1].at<double>(id,2)), 
                    Point3(AllPose[i-1].at<double>(id,3), AllPose[i-1].at<double>(id,4), AllPose[i-1].at<double>(id,5)));
                
            }
            // otherwise, get previous pose from last ping
            else
            {
                pose_dr_pre = Pose3(
                    Rot3::Rodrigues(AllPose[i].at<double>(j-1,0),AllPose[i].at<double>(j-1,1),AllPose[i].at<double>(j-1,2)), 
                    Point3(AllPose[i].at<double>(j-1,3), AllPose[i].at<double>(j-1,4), AllPose[i].at<double>(j-1,5)));
            }

            Pose3 odo = pose_dr_pre_tmp.between(pose_dr_cur);
            Pose3 odo_noise = add_noise.compose(odo);
            Pose3 pose_dr_cur_noisy = pose_dr_pre.compose(odo_noise);


            // update current pose with noise
            AllPose[i].at<double>(j,0) = pose_dr_cur_noisy.rotation().roll();
            AllPose[i].at<double>(j,1) = pose_dr_cur_noisy.rotation().pitch();
            AllPose[i].at<double>(j,2) = pose_dr_cur_noisy.rotation().yaw();
            AllPose[i].at<double>(j,3) = pose_dr_cur_noisy.x();
            AllPose[i].at<double>(j,4) = pose_dr_cur_noisy.y();
            AllPose[i].at<double>(j,5) = pose_dr_cur_noisy.z();

            // save pose_dr_cur to pre_tmp
            pose_dr_pre_tmp = pose_dr_cur;

        }
    }

    return;
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
