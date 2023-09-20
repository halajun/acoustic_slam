
#include "FEAmatcher.h"
#include <bits/stdc++.h>
#include "patchmatch.h"



namespace Diasss
{

using namespace std;
using namespace cv;

#define PI 3.14159265359

std::vector<cv::Mat> FEAmatcher::IniFlow(Frame &SourceFrame, Frame &TargetFrame)
{

    std::vector<cv::Mat> flow_st, flow_ts;

    cv::Mat img_s = SourceFrame.norm_img;
    cv::Mat img_t = TargetFrame.norm_img;

    cv::Mat flow_st_x = cv::Mat::zeros(img_s.rows, img_s.cols, CV_32FC1);
    cv::Mat flow_st_y = cv::Mat::zeros(img_s.rows, img_s.cols, CV_32FC1);

    cv::Mat ClusterMembers;
    std::vector<cv::Point2i> CMindices;
    ClusterMembers.create(cv::Size(2,img_s.rows*img_s.cols), CV_32FC1);
    int count = 0;
    for (size_t i = 0; i < img_s.rows; i++)
    {
        for (size_t j = 0; j < img_s.cols; j++)
        {
            CMindices.push_back(cv::Point2i(i,j));
            double x = SourceFrame.geo_img[0].at<double>(i,j);
            double y = SourceFrame.geo_img[1].at<double>(i,j);
            ClusterMembers.at<float>(count,0) = x;
            ClusterMembers.at<float>(count,1) = y;
            count++;
        }        
    }

    cv::Mat ClusterCenters;
    std::vector<cv::Point2i> CCindices;
    ClusterCenters.create(cv::Size(2,img_t.rows*img_t.cols), CV_32FC1);
    count = 0;
    for (size_t i = 0; i < img_t.rows; i++)
    {
        for (size_t j = 0; j < img_t.cols; j++)
        {
            CCindices.push_back(cv::Point2i(i,j));
            double x = TargetFrame.geo_img[0].at<double>(i,j);
            double y = TargetFrame.geo_img[1].at<double>(i,j);
            ClusterCenters.at<float>(count,0) = x;
            ClusterCenters.at<float>(count,1) = y;
            count++;
        }        
    }
    
    cv::Mat matches, distances;
    matches.create(cv::Size(1,img_s.rows*img_s.cols), CV_32SC1);
    distances.create(cv::Size(1,img_s.rows*img_s.cols), CV_32FC1);
    const cvflann::SearchParams params(32); //How many leaves to search in a tree
    cv::flann::GenericIndex< cvflann::L2<float> > *kdtrees; // The flann searching tree
    kdtrees =  new flann::GenericIndex< cvflann::L2<float> >(ClusterCenters, cvflann::KDTreeIndexParams(4)); // a 4 k-d tree

    // Search KdTree
    kdtrees->knnSearch(ClusterMembers, matches, distances, 1,  cvflann::SearchParams(8));
    int NN_index;
    float dist;
    for(int i = 0; i < img_s.rows*img_s.cols; i++)
    {
        NN_index = matches.at<int>(i,0);
        dist = distances.at<float>(i,0);

        if (SourceFrame.flt_mask.at<bool>(CMindices[i].x,CMindices[i].y) == 0)
            continue;

        if (TargetFrame.flt_mask.at<bool>(CCindices[NN_index].x,CCindices[NN_index].y) == 0)
            continue;

        // only focus on overlapped region
        if (dist>0.5)
        {
            // cout << dist << " ";
            continue;
        }
        

        if (false) // abs(SourceFrame.img_id-TargetFrame.img_id)%2!=0
        {
            flow_st_x.at<float>(CMindices[i].x,CMindices[i].y) = (img_t.rows - CCindices[NN_index].x - 1) - CMindices[i].x;  
            flow_st_y.at<float>(CMindices[i].x,CMindices[i].y) = (img_t.cols - CCindices[NN_index].y - 1) - CMindices[i].y;
            // cout << flow_st_x.at<float>(CMindices[i].x,CMindices[i].y) << "/" << flow_st_y.at<float>(CMindices[i].x,CMindices[i].y) << " ";
        }
        else
        {
            flow_st_x.at<float>(CMindices[i].x,CMindices[i].y) = CCindices[NN_index].x - CMindices[i].x;
            flow_st_y.at<float>(CMindices[i].x,CMindices[i].y) = CCindices[NN_index].y - CMindices[i].y;
        }

    }
    // cout << endl;

    delete kdtrees;

    flow_st.push_back(flow_st_y);
    flow_st.push_back(flow_st_x);
        
   
    return flow_st;
    

}

void FEAmatcher::DenseMatchingD(Frame &SourceFrame, Frame &TargetFrame)
{
    bool PLOT_IMG = 0;
    int count = 0, THRES_crosscheck = 3;

    // get normalized image;
    cv::Mat img_s = SourceFrame.norm_img;
    cv::Mat img_t = TargetFrame.norm_img;
    cv::Mat img_t_inv;
    if (abs(SourceFrame.img_id-TargetFrame.img_id)%2!=0)
        flip(TargetFrame.norm_img,img_t_inv,-1);

    // get initial flow from Dead-reckoning Prior
    cout << "(1) generating initial flow..." << endl;    
    std::vector<cv::Mat> ini_flow_st = FEAmatcher::IniFlow(SourceFrame,TargetFrame);
    cout << "halfway..." << endl;
    std::vector<cv::Mat> ini_flow_ts = FEAmatcher::IniFlow(TargetFrame,SourceFrame);
    cout << "completed!" << endl;

    // save flow to Img format
    struct Img odisp_st(img_s.cols, img_s.rows, 2);
    count = 0;
    for (size_t k = 0; k < ini_flow_st.size(); k++)
        for (size_t m = 0; m < ini_flow_st[k].rows; m++)
                for (size_t n = 0; n < ini_flow_st[k].cols; n++)
                {  
                odisp_st[count] = ini_flow_st[k].at<float>(m,n);
                count++;
                }
    struct Img odisp_ts(img_t.cols, img_t.rows, 2);
    count = 0;
    for (size_t k = 0; k < ini_flow_ts.size(); k++)
        for (size_t m = 0; m < ini_flow_ts[k].rows; m++)
                for (size_t n = 0; n < ini_flow_ts[k].cols; n++)
                {  
                odisp_ts[count] = ini_flow_ts[k].at<float>(m,n);
                count++;
                }  

    // save normalized image to Img format
    struct Img img_s_img(img_s.cols, img_s.rows);
    count = 0;
    for (size_t m = 0; m < img_s.rows; m++)
    {
        for (size_t n = 0; n < img_s.cols; n++)
        {
            img_s_img[count] = (float)img_s.at<uchar>(m,n);
            count++;
        }    
    }  
    struct Img img_t_img(img_t.cols, img_t.rows);
    count = 0;
    for (size_t m = 0; m < img_t.rows; m++)
    {
        for (size_t n = 0; n < img_t.cols; n++)
        {
            img_t_img[count] = (float)img_t.at<uchar>(m,n);
            count++;
        }    
    }


    // get final flow from patchmatch
    // --- (Img &u1, Img &u2, int w, Img &off, Img &cost,int minoff, int maxoff,  int iterations, int randomtrials, bool use_horizontal_off) --- //
    int patch_size = 13, minoff = 0, maxoff = 5, iterations = 10, randomtrials = 10;
    cout << "(2) processing patch matching..." << endl; 
    struct Img ocost_st(img_s.cols, img_s.rows);
    patchmatch<distance_patch_NCC>(img_s_img, img_t_img, patch_size, odisp_st, ocost_st, minoff, maxoff, iterations, randomtrials, false);
    cout << "halfway..." << endl;
    struct Img ocost_ts(img_t.cols, img_t.rows);
    patchmatch<distance_patch_NCC>(img_t_img, img_s_img, patch_size, odisp_ts, ocost_ts, minoff, maxoff, iterations, randomtrials, false);
    cout << "completed!" << endl;


    // --- demonstrate --- //
    std::vector<cv::DMatch> TemperalMatches;
    std::vector<cv::KeyPoint> PreKeys, CurKeys;
    count = 0;

    // find good correspondences (dense overlapping version)
    // nx for column, ny for row
    cv::Mat corrs_kps_tmp;
    for (int y=0;y<img_s_img.ny;y++) 
    {
        for (int x=0;x<img_s_img.nx;x++) 
        {

            const int r_offset = odisp_st(x,y,1);
            const int c_offset = odisp_st(x,y,0);

            if (r_offset==0 || c_offset==0)          
                continue;

            int y_posi_in_t = y+r_offset;
            int x_posi_in_t = x+c_offset;

            if (odisp_ts(x_posi_in_t,y_posi_in_t,1)==0 || odisp_ts(x_posi_in_t,y_posi_in_t,0)==0)         
                continue;

            const int y_prime = y_posi_in_t + odisp_ts(x_posi_in_t,y_posi_in_t,1);
            const int x_prime = x_posi_in_t + odisp_ts(x_posi_in_t,y_posi_in_t,0);

            if (y==y_prime && x==x_prime)
            {

                cv::Mat1i temp = (cv::Mat1i(1,6)<<  SourceFrame.img_id,TargetFrame.img_id,y,x,y_posi_in_t,x_posi_in_t);
                corrs_kps_tmp.push_back(temp); 

                PreKeys.push_back(cv::KeyPoint(x,y,0,0,0,-1));
                if (abs(SourceFrame.img_id-TargetFrame.img_id)%2!=0)
                    CurKeys.push_back(cv::KeyPoint(TargetFrame.norm_img.cols-x_posi_in_t-1,TargetFrame.norm_img.rows-y_posi_in_t-1,0,0,0,-1));
                else                
                    CurKeys.push_back(cv::KeyPoint(x_posi_in_t,y_posi_in_t,0,0,0,-1));
                TemperalMatches.push_back(cv::DMatch(count,count,0));
                count++;

            }       

        }
    }

    // --- SAVE RESULTS to IMG_S --- //
    SourceFrame.corres_kps_dense.push_back(corrs_kps_tmp);
    // SourceFrame.corres_kps_dense = corrs_kps_tmp; // corrs_kps_tmp.clone();

    // plot matches
    if (PLOT_IMG)
    {
        cout << "Good Matches Number: " << count << endl;
        cv::Mat img_matches;
        if (abs(SourceFrame.img_id-TargetFrame.img_id)%2!=0)
            cv::drawMatches(img_s, PreKeys, img_t_inv, CurKeys, TemperalMatches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                            vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        else
            cv::drawMatches(img_s, PreKeys, img_t, CurKeys, TemperalMatches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                            vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);                            
        cv::namedWindow("temperal matches", cv::WINDOW_NORMAL);
        cv::imshow("temperal matches", img_matches);
        cv::waitKey(0);
    }
    
    return;
}

void FEAmatcher::DenseMatchingS(Frame &SourceFrame, Frame &TargetFrame)
{
    bool PLOT_IMG = 1, METRIC_ERROR = false;
    int count = 0, THRES_crosscheck = 3;

    // get normalized image;
    cv::Mat img_s = SourceFrame.norm_img;
    cv::Mat img_t = TargetFrame.norm_img;
    cv::Mat img_t_inv;
    if (abs(SourceFrame.img_id-TargetFrame.img_id)%2!=0)
        flip(TargetFrame.norm_img,img_t_inv,-1);

    // get initial flow from Dead-reckoning Prior
    cout << "(1) generating initial flow..." << endl;    
    std::vector<cv::Mat> ini_flow_st = FEAmatcher::IniFlow(SourceFrame,TargetFrame);
    cout << "halfway..." << endl;
    std::vector<cv::Mat> ini_flow_ts = FEAmatcher::IniFlow(TargetFrame,SourceFrame);
    cout << "completed!" << endl;

    // save flow to Img format
    struct Img odisp_st(img_s.cols, img_s.rows, 2);
    struct Img odisp_st_ini(img_s.cols, img_s.rows, 2); // for comparison
    count = 0;
    for (size_t k = 0; k < ini_flow_st.size(); k++)
        for (size_t m = 0; m < ini_flow_st[k].rows; m++)
                for (size_t n = 0; n < ini_flow_st[k].cols; n++)
                {  
                odisp_st[count] = ini_flow_st[k].at<float>(m,n);
                odisp_st_ini[count] = ini_flow_st[k].at<float>(m,n);
                count++;
                }
    struct Img odisp_ts(img_t.cols, img_t.rows, 2);
    count = 0;
    for (size_t k = 0; k < ini_flow_ts.size(); k++)
        for (size_t m = 0; m < ini_flow_ts[k].rows; m++)
                for (size_t n = 0; n < ini_flow_ts[k].cols; n++)
                {  
                odisp_ts[count] = ini_flow_ts[k].at<float>(m,n);
                count++;
                }  

    // save normalized image to Img format
    struct Img img_s_img(img_s.cols, img_s.rows);
    count = 0;
    for (size_t m = 0; m < img_s.rows; m++)
    {
        for (size_t n = 0; n < img_s.cols; n++)
        {
            img_s_img[count] = (float)img_s.at<uchar>(m,n);
            count++;
        }    
    }  
    struct Img img_t_img(img_t.cols, img_t.rows);
    count = 0;
    for (size_t m = 0; m < img_t.rows; m++)
    {
        for (size_t n = 0; n < img_t.cols; n++)
        {
            img_t_img[count] = (float)img_t.at<uchar>(m,n);
            count++;
        }    
    }


    // get final flow from patchmatch
    // --- (Img &u1, Img &u2, int w, Img &off, Img &cost,int minoff, int maxoff,  int iterations, int randomtrials, bool use_horizontal_off) --- //
    int patch_size = 13, minoff = 0, maxoff = 5, iterations = 10, randomtrials = 10;
    cout << "(2) processing patch matching..." << endl; 
    struct Img ocost_st(img_s.cols, img_s.rows);
    patchmatch<distance_patch_NCC>(img_s_img, img_t_img, patch_size, odisp_st, ocost_st, minoff, maxoff, iterations, randomtrials, false);
    cout << "halfway..." << endl;
    struct Img ocost_ts(img_t.cols, img_t.rows);
    patchmatch<distance_patch_NCC>(img_t_img, img_s_img, patch_size, odisp_ts, ocost_ts, minoff, maxoff, iterations, randomtrials, false);
    cout << "completed!" << endl;


    // --- demonstrate --- //
    std::vector<cv::DMatch> TemperalMatches;
    std::vector<cv::KeyPoint> PreKeys, CurKeys;
    count = 0;

    // find good correspondences (annotated version)
    float e_r = 0, e_c = 0, e_rc = 0, e_rc_ini = 0, inl_num = 0;
    cv::Mat corrs_kps_tmp; //  = SourceFrame.anno_kps.clone()
    for (size_t n = 0; n < SourceFrame.anno_kps.rows; n=n+1)
    {
        if (SourceFrame.anno_kps.at<int>(n,1)!=TargetFrame.img_id)
            continue;
        
        const int r = SourceFrame.anno_kps.at<int>(n,2);
        const int c = SourceFrame.anno_kps.at<int>(n,3);
        const int r_gt = SourceFrame.anno_kps.at<int>(n,4);
        const int c_gt = SourceFrame.anno_kps.at<int>(n,5);

        const int r_offset = odisp_st(c,r,1);
        const int c_offset = odisp_st(c,r,0);
        const int r_offset_ini = odisp_st_ini(c,r,1);
        const int c_offset_ini = odisp_st_ini(c,r,0);


        if (r_offset==0 || c_offset==0)
        {
            cv::Mat1i temp = (cv::Mat1i(1,6)<<  SourceFrame.img_id,TargetFrame.img_id,r,c,-1,-1);
            corrs_kps_tmp.push_back(temp);            
            continue;
        }

        if (odisp_ts(c+c_offset,r+r_offset,1)==0 || odisp_ts(c+c_offset,r+r_offset,0)==0)
        {
            cv::Mat1i temp = (cv::Mat1i(1,6)<<  SourceFrame.img_id,TargetFrame.img_id,r,c,-1,-1);
            corrs_kps_tmp.push_back(temp);            
            continue;
        }
        
        const int r_prime = r + r_offset + odisp_ts(c+c_offset,r+r_offset,1);
        const int c_prime = c + c_offset + odisp_ts(c+c_offset,r+r_offset,0);

        // cout << "cross check (r/r',c/c'): " << r << "/" << r_prime << " , " << c << "/" << c_prime << endl;

        // save matches that satisfy cross check threshold
        if (abs(r-r_prime)<THRES_crosscheck && abs(c-c_prime)<THRES_crosscheck)
        {
            int r_posi_in_t = r+r_offset;
            int c_posi_in_t = c+c_offset;

            int r_posi_in_t_dr = r+r_offset_ini;
            int c_posi_in_t_dr = c+c_offset_ini;

            cv::Mat1i temp = (cv::Mat1i(1,6)<<  SourceFrame.img_id,TargetFrame.img_id,r,c,r_posi_in_t,c_posi_in_t);
            // cout << "check saved keypoints: " << temp.at<int>(0) << " " << temp.at<int>(1) << " " << temp.at<int>(2) << " " << temp.at<int>(3) << " " << temp.at<int>(4) << endl;
            corrs_kps_tmp.push_back(temp); 

            PreKeys.push_back(cv::KeyPoint(c,r,0,0,0,-1));
            if (abs(SourceFrame.img_id-TargetFrame.img_id)%2!=0)
                CurKeys.push_back(cv::KeyPoint(TargetFrame.norm_img.cols-c_posi_in_t-1,TargetFrame.norm_img.rows-r_posi_in_t-1,0,0,0,-1));
            else
                CurKeys.push_back(cv::KeyPoint(c+c_offset,r+r_offset,0,0,0,-1));
            TemperalMatches.push_back(cv::DMatch(count,count,0));
            count++;

            if (METRIC_ERROR)
            {
                const float e_rc_tmp = (r_gt - (r+r_offset))*(r_gt - (r+r_offset)) + (c_gt - (c+c_offset))*(c_gt - (c+c_offset));
                const float e_rc_tmp_ini = (r_gt - (r+r_offset_ini))*(r_gt - (r+r_offset_ini)) + (c_gt - (c+c_offset_ini))*(c_gt - (c+c_offset_ini));
                cout << "error per point (r c sr): " << abs(r_gt - (r+r_offset_ini)) << "->" << abs(r_gt - (r+r_offset)) << " ";
                cout << abs(c_gt - (c+c_offset_ini)) << "->" << abs(c_gt - (c+c_offset)) << " " << sqrt(e_rc_tmp_ini) << "->" << sqrt(e_rc_tmp) << endl;

                if (sqrt(e_rc_tmp)<=1.2*sqrt(e_rc_tmp_ini) || (sqrt(e_rc_tmp)<6))
                    inl_num = inl_num + 1.0;

                e_r = e_r + abs(r_gt - (r+r_offset));
                e_c = e_c + abs(c_gt - (c+c_offset));
                e_rc = e_rc + e_rc_tmp;   
                e_rc_ini = e_rc_ini + e_rc_tmp_ini; 
            }
              

        }
        else
        {
            cv::Mat1i temp = (cv::Mat1i(1,6)<<  SourceFrame.img_id,TargetFrame.img_id,r,c,-1,-1);
            corrs_kps_tmp.push_back(temp);              
        }        

    }

    // SAVE RESULTS to IMG_S
    SourceFrame.corres_kps.push_back(corrs_kps_tmp);

    // metric matching error
    if (METRIC_ERROR)
    {
        e_r = e_r/count;
        e_c = e_c/count;
        e_rc = sqrt(e_rc/count);
        e_rc_ini = sqrt(e_rc_ini/count);
        cout << "Metric Error between " << SourceFrame.img_id << " and " << TargetFrame.img_id << " (r c rc rc_ini %): ";
        cout << e_r << " " << e_c << " " << e_rc << " " << e_rc_ini << " " << inl_num/count*100 << "%" << endl;
    }
    
    // plot matches
    if (PLOT_IMG)
    {
        cout << "Good Matches Number: " << count << endl;
        cv::Mat img_matches;
        if (abs(SourceFrame.img_id-TargetFrame.img_id)%2!=0)
            cv::drawMatches(img_s, PreKeys, img_t_inv, CurKeys, TemperalMatches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                            vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        else
            cv::drawMatches(img_s, PreKeys, img_t, CurKeys, TemperalMatches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                            vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);  
        cv::namedWindow("temperal matches", cv::WINDOW_NORMAL);
        cv::imshow("temperal matches", img_matches);
        cv::waitKey(0);
    }

    PreKeys.clear();
    CurKeys.clear();
    TemperalMatches.clear();
    count = 0;

    // find good correspondences (annotated version)
    cv::Mat corrs_kps_tmp2; //  = SourceFrame.anno_kps.clone()
    for (size_t n = 0; n < TargetFrame.anno_kps.rows; n=n+1)
    {
        if (TargetFrame.anno_kps.at<int>(n,1)!=SourceFrame.img_id)
            continue;
        
        const int r = TargetFrame.anno_kps.at<int>(n,2);
        const int c = TargetFrame.anno_kps.at<int>(n,3);

        const int r_offset = odisp_ts(c,r,1);
        const int c_offset = odisp_ts(c,r,0);

        if (r_offset==0 || c_offset==0)
        {
            cv::Mat1i temp = (cv::Mat1i(1,6)<<  TargetFrame.img_id,SourceFrame.img_id,r,c,-1,-1);
            corrs_kps_tmp2.push_back(temp);            
            continue;
        }
        
        if (odisp_st(c+c_offset,r+r_offset,1)==0 || odisp_st(c+c_offset,r+r_offset,0)==0)
        {
            cv::Mat1i temp = (cv::Mat1i(1,6)<<  TargetFrame.img_id,SourceFrame.img_id,r,c,-1,-1);
            corrs_kps_tmp2.push_back(temp);            
            continue;
        }
        
        const int r_prime = r + r_offset + odisp_st(c+c_offset,r+r_offset,1);
        const int c_prime = c + c_offset + odisp_st(c+c_offset,r+r_offset,0);

        // cout << "cross check (r/r',c/c'): " << r << "/" << r_prime << " , " << c << "/" << c_prime << endl;

        // save matches that satisfy cross check threshold
        if (abs(r-r_prime)<THRES_crosscheck && abs(c-c_prime)<THRES_crosscheck)
        {
            int r_posi_in_s = r+r_offset;
            int c_posi_in_s = c+c_offset;
            
            if (false) // abs(SourceFrame.img_id-TargetFrame.img_id)%2!=0
            {
                r_posi_in_s = TargetFrame.norm_img.rows-r_posi_in_s-1;
                c_posi_in_s = TargetFrame.norm_img.cols-c_posi_in_s-1;
            }

            cv::Mat1i temp = (cv::Mat1i(1,6)<<  TargetFrame.img_id,SourceFrame.img_id,r,c,r_posi_in_s,c_posi_in_s);
            corrs_kps_tmp2.push_back(temp); 

            PreKeys.push_back(cv::KeyPoint(c,r,0,0,0,-1));                
            CurKeys.push_back(cv::KeyPoint(c+c_offset,r+r_offset,0,0,0,-1));
            TemperalMatches.push_back(cv::DMatch(count,count,0));
            count++;  

        }
        else
        {
            cv::Mat1i temp = (cv::Mat1i(1,6)<<  TargetFrame.img_id,SourceFrame.img_id,r,c,-1,-1);
            corrs_kps_tmp2.push_back(temp);              
        }       

    }

    // SAVE RESULTS to IMG_T
    TargetFrame.corres_kps.push_back(corrs_kps_tmp2);

    // plot matches
    if (PLOT_IMG)
    {
        cout << "Good Matches Number: " << count << endl;
        cv::Mat img_matches;
        cv::drawMatches(img_t, PreKeys, img_s, CurKeys, TemperalMatches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                        vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::namedWindow("temperal matches", cv::WINDOW_NORMAL);
        cv::imshow("temperal matches", img_matches);
        cv::waitKey(0);
    }


    return;
}

void FEAmatcher::RobustMatching(Frame &SourceFrame, Frame &TargetFrame)
{

    // std::vector<std::pair<size_t, size_t> > CorresPairs;

    std::vector<cv::KeyPoint> kps_1 = SourceFrame.kps;
    std::vector<cv::KeyPoint> kps_2 = TargetFrame.kps;
    cv::Mat dst_1 = SourceFrame.dst;
    cv::Mat dst_2 = TargetFrame.dst;
    std::vector<cv::Mat> geo_img_1 = SourceFrame.geo_img;
    std::vector<cv::Mat> geo_img_2 = TargetFrame.geo_img;
    std::vector<std::pair<int,double>> scc_1, scc_2;

    std::vector<int> CorresID_1 = FEAmatcher::GeoNearNeighSearch(SourceFrame.img_id,TargetFrame.img_id,SourceFrame.norm_img,TargetFrame.norm_img,
                                                                 kps_1,dst_1,geo_img_1,kps_2,dst_2,geo_img_2, scc_1);
    std::vector<int> CorresID_2 = FEAmatcher::GeoNearNeighSearch(TargetFrame.img_id,SourceFrame.img_id,TargetFrame.norm_img,SourceFrame.norm_img,
                                                                 kps_2,dst_2,geo_img_2,kps_1,dst_1,geo_img_1, scc_2);

    std::vector<cv::KeyPoint> SourceKeys, TargetKeys;
    FEAmatcher::ConsistentCheck(SourceFrame,TargetFrame,CorresID_1,CorresID_2,scc_1,scc_2,SourceKeys,TargetKeys);

    // save matches to Frame
    for (size_t i = 0; i < SourceKeys.size(); i++)
    {
        cv::Mat1d kp_pair_s = (cv::Mat1d(1,6)<<SourceFrame.img_id,TargetFrame.img_id,
                                               SourceKeys[i].pt.y,SourceKeys[i].pt.x,
                                               TargetKeys[i].pt.y,TargetKeys[i].pt.x);
        SourceFrame.corres_kps.push_back(kp_pair_s);
        cv::Mat1d kp_pair_t = (cv::Mat1d(1,6)<<TargetFrame.img_id,SourceFrame.img_id,
                                               TargetKeys[i].pt.y,TargetKeys[i].pt.x,
                                               SourceKeys[i].pt.y,SourceKeys[i].pt.x);
        TargetFrame.corres_kps.push_back(kp_pair_t);        
    }
    

    return;

}

std::vector<int> FEAmatcher::GeoNearNeighSearch(const int &img_id, const int &img_id_ref,
                                                const cv::Mat &img, const cv::Mat &img_ref,
                                                const std::vector<cv::KeyPoint> &kps, const cv::Mat &dst, const std::vector<cv::Mat> &geo_img,
                                                const std::vector<cv::KeyPoint> &kps_ref, const cv::Mat &dst_ref, const std::vector<cv::Mat> &geo_img_ref,
                                                std::vector<std::pair<int,double>> &scc)
{
    // cv::RNG rng((unsigned)time(NULL));
    cv::RNG rng;
    cv::setRNGSeed(1);
    std::vector<int> CorresID = std::vector<int>(kps.size(),-1);
    std::vector<int> ID_loc;
    bool USE_SIFT = 1, SCC_x = 1, SCC_xy = 0;

    // --- some parameters --- //
    int radius = 8; // search circle size
    int x_min, y_min, x_max, y_max; // rectangle of seach window
    double bx_min, bx_max, by_min, by_max; // geometric border of reference image

    // get boundary of the reference geo-image
    cv::minMaxLoc(geo_img_ref[0], &bx_min, &bx_max);
    cv::minMaxLoc(geo_img_ref[1], &by_min, &by_max);

    // cout << "boundary: " << bx_min << " " << bx_max << " " << by_min << " " << by_max << endl;

    // --- main loop --- //
    vector<int> candidate;
    std::vector<cv::KeyPoint> kps_show, kps_show_ref;
    for (size_t i = 0; i < kps.size(); i++)
    {
        double loc_x = geo_img[0].at<double>(kps[i].pt.y,kps[i].pt.x);
        double loc_y = geo_img[1].at<double>(kps[i].pt.y,kps[i].pt.x);

        if (loc_x<bx_min || loc_y<by_min || loc_x>bx_max || loc_y>by_max)
            continue;
        
        for (size_t j = 0; j < kps_ref.size(); j++)
        {
            double ref_loc_x = geo_img_ref[0].at<double>(kps_ref[j].pt.y,kps_ref[j].pt.x);
            double ref_loc_y = geo_img_ref[1].at<double>(kps_ref[j].pt.y,kps_ref[j].pt.x);

            double geo_dist = sqrt( (loc_x-ref_loc_x)*(loc_x-ref_loc_x) + (loc_y-ref_loc_y)*(loc_y-ref_loc_y) );
            if (geo_dist<radius)
            {
                candidate.push_back(j);
                kps_show_ref.push_back(kps_ref[j]); 
            }          
        }

        // cout << "candidate size: " << candidate.size() << endl;

        if(!candidate.size())
            continue;

        // --- using SIFT --- //
        if (USE_SIFT)
        {
            double best_dist = 1000, sec_best_dist = 1000, dist_bound = 350; // 150
            int best_id = -1;
            double ratio_test = 0.35; // 0.5
            for (size_t j = 0; j < candidate.size(); j++)
            {
                const double dst_dist = cv::norm(dst.row(i),dst_ref.row(candidate[j]),cv::NORM_L2);
                // cout << "dist: " << dst_dist << endl;
                if (dst_dist<best_dist)
                {
                    sec_best_dist = best_dist;
                    best_dist = dst_dist;
                    best_id = candidate[j];
                }
                else if (dst_dist<sec_best_dist)
                {
                  sec_best_dist = dst_dist;
                }
                
            }
            double fir_sec_ratio = best_dist/sec_best_dist;
            // cout << "best and second best ratio: " << fir_sec_ratio  << " " << best_dist << " " << sec_best_dist << endl;
            if (best_id!=-1 && best_dist<dist_bound && fir_sec_ratio<=ratio_test)
            {
                CorresID[i] = best_id;
                ID_loc.push_back(i);
            }
            else if (candidate.size()==1 && best_dist<dist_bound)
            {
                CorresID[i] = best_id;
                ID_loc.push_back(i);
            }
        }
        // --- using ORB --- //
        else
        {
            int best_dist = 1000, sec_best_dist = 1000, dist_bound = 88; // 88
            if (img_id%2!=img_id_ref%2)
                dist_bound = 80; // 80           
            int best_id = -1;
            double ratio_test = 0.35; // 0.35
            for (size_t j = 0; j < candidate.size(); j++)
            {
                const int dst_dist = FEAmatcher::DescriptorDistance(dst.row(i),dst_ref.row(candidate[j]));
                // cout << "dist: " << dst_dist << endl;
                if (dst_dist<best_dist)
                {
                    sec_best_dist = best_dist;
                    best_dist = dst_dist;
                    best_id = candidate[j];
                }
                else if (dst_dist<sec_best_dist)
                {
                  sec_best_dist = dst_dist;
                }
                
            }
            double fir_sec_ratio = (double)best_dist/sec_best_dist;
            // cout << "best and second best ratio: " << fir_sec_ratio  << " " << best_dist << " " << sec_best_dist << endl;
            if (best_id!=-1 && best_dist<=dist_bound && fir_sec_ratio<=ratio_test && sec_best_dist!=1000)
            {
                CorresID[i] = best_id;
                ID_loc.push_back(i);
            }
            else if (candidate.size()==1 && best_dist<=dist_bound)
            {
                CorresID[i] = best_id;
                ID_loc.push_back(i);
            }
        }
            
             
        candidate.clear();
        kps_show.clear();
        kps_show_ref.clear();

    }

    // --- Sliding Compatibility Check (SCC) on the X axis of keypoints --- //
    if (SCC_x)
    {
        // cout << "size of first selections: " << ID_loc.size() << endl;
        int final_inlier_num = 0, iter_num = 0, max_iter = 1000, sam_num = 2;
        double PixError = 2.5; // 2.5
        std::vector<int> CorresID_final = std::vector<int>(kps.size(),-1);
        while (iter_num<max_iter)
        {
            int cur_inlier_num = 0;
            std::vector<int> CorresID_iter = std::vector<int>(kps.size(),-1);

            // random sample matched ID from CorresID
            vector<int> sampled_ids(sam_num,0);
            for (size_t i = 0; i < sam_num; i++)
            {
                const int getID = rng.uniform(0,ID_loc.size());
                sampled_ids[i]=ID_loc[getID];
                // cout << sampled_ids[i] << " ";
            }
            // calculate model X
            double ModelX = 0;
            for (size_t i = 0; i < sampled_ids.size(); i++)
            {
                if (img_id%2!=img_id_ref%2)
                    ModelX = ModelX + abs(kps[sampled_ids[i]].pt.y - (img_ref.rows-kps_ref[CorresID[sampled_ids[i]]].pt.y+1));
                else
                    ModelX = ModelX + abs(kps[sampled_ids[i]].pt.y - kps_ref[CorresID[sampled_ids[i]]].pt.y);
            }
            ModelX = ModelX/sam_num;
            // fit all CorresID to Model, and find inliers
            for (size_t j = 0; j < CorresID.size(); j++)
            {
                if (CorresID[j]==-1)
                    continue;
        
                double X_tmp;
                if (img_id%2!=img_id_ref%2)
                {
                    X_tmp= abs(kps[j].pt.y - (img_ref.rows-kps_ref[CorresID[j]].pt.y+1));
                }
                else
                    X_tmp= abs(kps[j].pt.y - kps_ref[CorresID[j]].pt.y);
                    
                // cout << "distance: " << abs(ModelX-X_tmp) << endl;
                if (abs(ModelX-X_tmp)<=PixError)
                {
                    CorresID_iter[j] = CorresID[j];
                    cur_inlier_num = cur_inlier_num + 1;
                }          
            }
            // update the most inlier set
            if (final_inlier_num<cur_inlier_num)
            {
                CorresID_final = CorresID_iter;
                final_inlier_num = cur_inlier_num;
                scc.push_back(std::make_pair(cur_inlier_num,ModelX)); 
            }
            iter_num = iter_num + 1;
        }
        cout << "initial inlier number: " << CorresID.size()-std::count(CorresID.begin(), CorresID.end(), -1) << endl;
        CorresID = CorresID_final;
        cout << "final inlier number: " << CorresID.size()-std::count(CorresID.begin(), CorresID.end(), -1) << endl;
    }

    // --- Sliding Compatibility Check (SCC) on the X  and Y axis of keypoints --- //
    if (SCC_xy)
    {
        // cout << "size of first selections: " << ID_loc.size() << endl;
        int final_inlier_num = 0, iter_num = 0, max_iter = 1000, sam_num = 3;
        double PixError_X =2.5, PixError_Y = 15.0; // 3.0, 3.0
        std::vector<int> CorresID_final = std::vector<int>(kps.size(),-1);
        while (iter_num<max_iter)
        {
            int cur_inlier_num = 0;
            std::vector<int> CorresID_iter = std::vector<int>(kps.size(),-1);

            // random sample matched ID from CorresID
            vector<int> sampled_ids(sam_num,0);
            for (size_t i = 0; i < sam_num; i++)
            {
                const int getID = rng.uniform(0,ID_loc.size());
                sampled_ids[i]=ID_loc[getID];
                // cout << sampled_ids[i] << " ";
            }
            // calculate model X and Y
            double ModelX = 0, ModelY = 0;
            for (size_t i = 0; i < sampled_ids.size(); i++)
            {
                if (img_id%2!=img_id_ref%2)
                    ModelX = ModelX + abs(kps[sampled_ids[i]].pt.y - (img_ref.rows-kps_ref[CorresID[sampled_ids[i]]].pt.y+1));
                else
                    ModelX = ModelX + abs(kps[sampled_ids[i]].pt.y - kps_ref[CorresID[sampled_ids[i]]].pt.y);

                ModelY = ModelY + abs(kps[sampled_ids[i]].pt.x - kps_ref[CorresID[sampled_ids[i]]].pt.x);
            }
            ModelX = ModelX/sam_num;
            ModelY = ModelY/sam_num;
            // cout << "MODEL Y: " << ModelY << endl;
            // fit all CorresID to Model, and find inliers
            for (size_t j = 0; j < CorresID.size(); j++)
            {
                if (CorresID[j]==-1)
                    continue;
        
                double X_tmp, Y_tmp;
                if (img_id%2!=img_id_ref%2)
                    X_tmp= abs(kps[j].pt.y - (img_ref.rows-kps_ref[CorresID[j]].pt.y+1));
                else
                    X_tmp= abs(kps[j].pt.y - kps_ref[CorresID[j]].pt.y);
                Y_tmp= abs(kps[j].pt.x - kps_ref[CorresID[j]].pt.x);
                    
                // cout << "X distance: " << abs(ModelX-X_tmp) << endl;
                // cout << "Y distance: " << abs(ModelY-Y_tmp) << endl;
                if (abs(ModelX-X_tmp)<=PixError_X && abs(ModelY-Y_tmp)<=PixError_Y)
                {
                    CorresID_iter[j] = CorresID[j];
                    cur_inlier_num = cur_inlier_num + 1;
                }          
            }
            // update the most inlier set
            if (final_inlier_num<cur_inlier_num)
            {
                CorresID_final = CorresID_iter;
                final_inlier_num = cur_inlier_num;
                scc.push_back(std::make_pair(cur_inlier_num,ModelX)); 
            }
            iter_num = iter_num + 1;
        }
        cout << "initial inlier number: " << CorresID.size()-std::count(CorresID.begin(), CorresID.end(), -1) << endl;
        CorresID = CorresID_final;
        cout << "final inlier number: " << CorresID.size()-std::count(CorresID.begin(), CorresID.end(), -1) << endl;
    }
    
  
    return CorresID;
}

void FEAmatcher::ConsistentCheck(const Frame &SourceFrame, const Frame &TargetFrame,
                                 const std::vector<int> &CorresID_1,const std::vector<int> &CorresID_2,
                                 std::vector<std::pair<int,double>> &scc_1, std::vector<std::pair<int,double>> &scc_2,
                                 std::vector<cv::KeyPoint> &SourceKeys, std::vector<cv::KeyPoint> &TargetKeys)
{
    bool show_match = 0;
    double kp_diff_thres = 2.5;

    std::sort(scc_1.rbegin(), scc_1.rend());
    std::sort(scc_2.rbegin(), scc_2.rend());
    // for (size_t i = 0; i < 3; i++)
    //     cout << "scc_1: " << scc_1[i].first << " " << scc_1[i].second << " " << abs(SourceFrame.norm_img.rows-TargetFrame.norm_img.rows) << endl;
    // for (size_t i = 0; i < 3; i++)
    //     cout << "scc_2: " << scc_2[i].first << " " << scc_2[i].second << endl;
    
    std::vector<cv::DMatch> TemperalMatches;
    int count = 0;
    // --- merge if the sliding compatibility check is consistent --- //
    double img_diff = 0;
    if (SourceFrame.img_id%2!=TargetFrame.img_id%2)
        img_diff = abs(SourceFrame.norm_img.rows-TargetFrame.norm_img.rows);    
    double kp_diff = abs(abs(scc_1[0].second-scc_2[0].second)-img_diff);
    if (kp_diff<=kp_diff_thres)
    {
        // cout << "kp_diff: " << kp_diff << endl;
        for (size_t i = 0; i < CorresID_1.size(); i++)
        {
            if (CorresID_1[i]==-1)
                continue;

            if (CorresID_2[CorresID_1[i]]==i)
                continue;

            SourceKeys.push_back(SourceFrame.kps[i]);
            TargetKeys.push_back(TargetFrame.kps[CorresID_1[i]]);
            TemperalMatches.push_back(cv::DMatch(count,count,0));
            count = count + 1;    
        }

        for (size_t i = 0; i < CorresID_2.size(); i++)
        {
            if (CorresID_2[i]==-1)
                continue;
                
            SourceKeys.push_back(SourceFrame.kps[CorresID_2[i]]);
            TargetKeys.push_back(TargetFrame.kps[i]);
            TemperalMatches.push_back(cv::DMatch(count,count,0));
            count = count + 1;    
        }       
    }
    // --- otherwise, keep the matched pairs from the direction with more pairs --- //
    else
    {
        const int inl_num_1 = CorresID_1.size()-std::count(CorresID_1.begin(), CorresID_1.end(), -1);
        const int inl_num_2 = CorresID_2.size()-std::count(CorresID_2.begin(), CorresID_2.end(), -1);
        if (inl_num_1>inl_num_2)
        {
            for (size_t i = 0; i < CorresID_1.size(); i++)
            {
                if (CorresID_1[i]==-1)
                    continue;
                    
                SourceKeys.push_back(SourceFrame.kps[i]);
                TargetKeys.push_back(TargetFrame.kps[CorresID_1[i]]);
                TemperalMatches.push_back(cv::DMatch(count,count,0));
                count = count + 1;    
            }
        }
        else
        {
            for (size_t i = 0; i < CorresID_2.size(); i++)
            {
                if (CorresID_2[i]==-1)
                    continue;
                    
                SourceKeys.push_back(SourceFrame.kps[CorresID_2[i]]);
                TargetKeys.push_back(TargetFrame.kps[i]);
                TemperalMatches.push_back(cv::DMatch(count,count,0));
                count = count + 1;    
            }
        }
            
    }

    // // --- cross-check --- //
    // std::vector<cv::DMatch> TemperalMatches;
    // int count = 0;
    // for (size_t i = 0; i < CorresID_1.size(); i++)
    // {
    //     if (CorresID_1[i]==-1)
    //         continue;

    //     if (CorresID_2[CorresID_1[i]]==i)
    //     {
    //         SourceKeys.push_back(SourceFrame.kps[i]);
    //         TargetKeys.push_back(TargetFrame.kps[CorresID_1[i]]);
    //         TemperalMatches.push_back(cv::DMatch(count,count,0));
    //         count = count + 1;
    //     }     
    // }
    
    cout << "===> cross check number: " << SourceKeys.size() << endl;

    // --- demonstrate --- //
    if (show_match)
    {
        cv::Mat img_matches;
        cv::drawMatches(SourceFrame.norm_img, SourceKeys, TargetFrame.norm_img, TargetKeys,
                    TemperalMatches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                    vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::namedWindow("temperal matches", cv::WINDOW_NORMAL);
        cv::imshow("temperal matches", img_matches);
        cv::waitKey(0); 
    } 


}

// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int FEAmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}


} // namespace Diasss
