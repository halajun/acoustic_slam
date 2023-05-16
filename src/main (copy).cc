/* Copyright (C) 2016, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>*/
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include <cmath> // std::abs for float and double
#include <numeric>
#include <vector>
#include <iostream>

#include "point.h"
#ifdef _OPENMP 
#include <omp.h>
#endif //_OPENMP

//// a structure to wrap images 
#include "img.h"

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

/************ IMG IO  **************/

extern "C" {
#include "iio.h"
}

struct Img iio_read_vector_split(char *nm)
{
	struct Img out;
	float *tmpout = iio_read_image_float_split(nm, &out.nx, &out.ny, &out.nch);
	out.data.assign(tmpout,tmpout + out.nx * out.ny * out.nch);
	out.npix = out.nx * out.ny;
	free (tmpout);
	return out;
}

void iio_write_vector_split(char *nm, struct Img &out)
{
	// .front() -> .data() in C++11
	iio_save_image_float_split(nm, &(out.data.front()), out.nx, out.ny, out.nch);
}


void remove_nonfinite_values_Img(struct Img &u, float newval) 
{
   for(int i=0;i<u.npix*u.nch;i++) 
      if (!std::isfinite(u[i])) u[i] = newval; 
}

// c: pointer to original argc
// v: pointer to original argv
// o: option name after hyphen
// d: default value (if NULL, the option takes no argument)
static char *pick_option(int *c, char ***v, char *o, char *d)
{
   int argc = *c;
   char **argv = *v;
   int id = d ? 1 : 0;
   for (int i = 0; i < argc - id; i++)
      if (argv[i][0] == '-' && 0 == strcmp(argv[i] + 1, o)) {
	 char *r = argv[i + id] + 1 - id;
	 *c -= id + 1;
	 for (int j = i; j < argc - id; j++)
	    (*v)[j] = (*v)[j + id + 1];
	 return r;
      }
   return d;
}

/************ Patch Match Functions  **************/

inline int check_inside_image(const int x, const int y, const struct Img &u) 
{
	if(x>=0 && y>=0 && x<u.nx && y<u.ny) return 1;
	else return 0;
}

inline float valnan(const struct Img &u, const int px, const int py, const int ch=0)
{
	return check_inside_image(px, py, u) ? u(px, py, ch) : NAN;
}

inline float valneumann(const struct Img &u, const int x, const int y, const int ch=0)
{  
   int xx=x, yy=y;
   xx = x >=  0  ? xx : 0;
   xx = x < u.nx ? xx : u.nx - 1;
   yy = y >=  0  ? yy : 0;
   yy = y < u.ny ? yy : u.ny - 1;
	return u(xx,yy,ch);
}

int RANDOMTRIALS=10;

inline float distance_patch_SxD(const Img &p1, const Img &p2, const float &curr_cost, const int type)
{
   float dist = 0.;
   for(int i=0;i<p1.npix*p1.nch;i++){
      float d = p1[i]-p2[i];
      dist += (type == 1 ? std::abs(d) : d*d);
//      if(dist > curr_cost) return dist;
   }
   return dist;
}

inline float distance_patch_SSD(const Img &p1, const Img &p2, const float &curr_cost)
{
   return distance_patch_SxD(p1, p2, curr_cost, 2);
}

inline float distance_patch_SAD(const Img &p1, const Img &p2, const float &curr_cost)
{
   return distance_patch_SxD(p1, p2, curr_cost, 1);
}

inline float distance_patch_ZSxD(const Img &p1, const Img &p2, const float &curr_cost, const int type)
{
   float dist = 0.;
   for(int c=0;c<p1.nch;c++){
      float mu1=0, mu2=0;
      for(int i=0;i<p1.npix;i++){
         mu1 += p1[i+p1.npix*c];
         mu2 += p2[i+p1.npix*c];
      }
      mu1/=p1.npix;
      mu2/=p1.npix;
      for(int i=0;i<p1.npix;i++){
         float d = std::abs(p1[i]-mu1-(p2[i]-mu2));
         dist += (type == 1 ? std::abs(d) : d*d);
//         if(dist > curr_cost) return dist;
      }
   }
   return dist;
}

inline float distance_patch_ZSSD(const Img &p1, const Img &p2, const float &curr_cost)
{
   return distance_patch_ZSxD(p1, p2, curr_cost, 2);
}

inline float distance_patch_ZSAD(const Img &p1, const Img &p2, const float &curr_cost)
{
   return distance_patch_ZSxD(p1, p2, curr_cost, 1);
}

inline float distance_patch_NCC(const Img &p1, const Img &p2, const float &curr_cost)
{
   float dist = 0.;
   for(int c=0;c<p1.nch;c++){
      int choffset = p1.npix*c;
      float mu1=0, mu2=0;        // means
      for(int i=0;i<p1.npix;i++){
         mu1 += p1[i+choffset];
         mu2 += p2[i+choffset];
      }
      mu1/=p1.npix;
      mu2/=p1.npix;
      double sig1=0, sig2=0;     // variances
      double distl = 0.;
      for(int i=0;i<p1.npix;i++){
         float d1 = p1[i+choffset]-mu1;
         float d2 = p2[i+choffset]-mu2;
         float d = (d1-d2);
         sig1  += d1*d1; 
         sig2  += d2*d2; 
         distl += d*d;
      }
      dist += distl/(sqrt(sig1)*sqrt(sig2));
      if(dist > curr_cost) return dist;
   }
   return dist;
}

// ker is just needed for the size of the patch
// extract a patch where the channels are planes
inline void extract_patch_integer(const Img &u, int x, int y, int knc, int knr , Img &p)
{
    int halfknc = knc/2;
    int halfknr = knr/2;
    int nch = u.nch;
    int nc  = u.ncol;
    int nr  = u.nrow;
    int a,b,c,i=0;

    for (c=0;c<nch;c++)
        for (b=0;b<knr;b++)
            for (a=0;a<knc;a++)
            {
                 p[i++] = valneumann(u,x+a-halfknc,y+b-halfknr,c);
            }
}

// ker is just needed for the size of the patch
// extract a patch where the channels are planes
inline void extract_patch_integer_noboundary(const Img &u, const int x, const int y, int knc, int knr , Img &p)
{
    const int halfknc = knc/2;
    const int halfknr = knr/2;
    const int nch = u.nch;
    const int nc  = u.ncol;
    const int nr  = u.nrow;
    int a,b,c,i=0;
    
    for (c=0;c<nch;c++) {
        int pu = nc*nr*c;
        
        for (b=0;b<knr;b++) {
            int ppu = pu + nc*(y+b-halfknr);
            
            ppu += x - halfknc;
            for (a=0;a<knc;a++) {
                p[i++] = u[ppu++];
            }
        }
    }
}


// ker is just needed for the size of the patch
// extract a patch where the channels are planes
inline void extract_patch_secure(const Img &u, const float x, const float y, Img &p)
{
    const int knc = p.ncol, knr = p.nrow;
    const int halfknc = knc/2;
    const int halfknr = knr/2;
    const int nch = u.nch;
    const int nc  = u.ncol;
    const int nr  = u.nrow;
    
    if ((x-(int)x == 0) && (y-(int)y == 0)) {
        if(x-halfknc<0 || x+halfknc>=nc || y-halfknr<0 || y+halfknr>=nr)
            return extract_patch_integer(u, (int) x, (int) y, knc, knr, p);
        else
            return extract_patch_integer_noboundary(u, (int) x, (int) y, knc, knr, p);
    }
    
    // still integer
    return extract_patch_integer(u, (int) x, (int) y, knc, knr, p);
}


typedef float(*patch_distance_func)(const Img &, const Img &, const float&); //signature of all patch distance functions

template<patch_distance_func dist>
void random_search(Img &u1, Img &u2, int w, Img &off, Img &cost, int minoff, int maxoff, bool use_horizontal_off)
{
    int maxtrials = RANDOMTRIALS;
   //  int maxtrials = 1;
    int thmax = 1; // thread handling data
    #ifdef _OPENMP 
    thmax = omp_get_max_threads();
    #endif
    std::vector<Img> p1(thmax, Img(w,w,u1.nch)), 
                     p2(thmax, Img(w,w,u1.nch)); 
    // random seeds
    static std::vector<unsigned int> seeds(thmax);
    for(int i=0; i<thmax; i++) seeds[i]+=i;

#pragma omp parallel for shared(p1,p2) 
    for (int y=0;y<u1.ny;y++) {
        for (int x=0;x<u1.nx;x++) {
            int thid = 0;
            #ifdef _OPENMP // thread id
            thid = omp_get_thread_num();
            #endif
            PointD2 curr(off(x,y,0), off(x,y,1));
            PointD2 updt(off(x,y,0), off(x,y,1));
            if(curr[0]==0 || curr[1]==0) continue;
            
            float curr_cost = cost(x,y);
            extract_patch_secure(u1, x, y, p1[thid]);
            
            for (int trial=0;trial<maxtrials; trial++) {
                PointD2 tmpoff( ((int) ( ( ( (double) rand_r(&seeds[thid]) ) / ((double) RAND_MAX + 1.0) ) *2* maxoff) - maxoff) + curr[0],
                           ((int) ( ( ( (double) rand_r(&seeds[thid]) ) / ((double) RAND_MAX + 1.0) ) *2* maxoff) - maxoff) * (1-use_horizontal_off) + curr[1] );
               //  cout << "ramdom search: " << off[0] << " " << off[1] << endl;
                
                PointD2 tmp = PointD2(x,y) + tmpoff;
                // skip points that fell outside the image or offsets smaller than minoff 
                if( hypot(tmpoff[0],tmpoff[1]) < minoff || 
                    ! check_inside_image(tmp[0],tmp[1], u2)  ) continue;
                
                extract_patch_secure(u2, tmp[0], tmp[1], p2[thid]);
                float new_cost = dist(p1[thid], p2[thid], curr_cost);
                
                if( new_cost < curr_cost ) {
                  //   cout << "current and new cost: " << curr_cost << "/" << new_cost << endl;
                  //   cout << "new offset: " << tmpoff[0] << "/" << tmpoff[1]  << endl;
                    curr_cost=new_cost;
                    updt = tmpoff;
                }
                
            }
            cost(x,y) = curr_cost;
            off(x,y,0) = updt[0];
            off(x,y,1) = updt[1];
        }
    }
}

template<patch_distance_func dist>
void propagation(Img &u1, Img &u2, int w, Img &off, Img &cost, int minoff, int maxoff, const int direction)
{
    int maxtrials = RANDOMTRIALS;
    int thmax = 1; // thread handling data
    #ifdef _OPENMP 
    thmax = omp_get_max_threads();
    #endif
    std::vector<Img> p1(thmax, Img(w,w,u1.nch)), 
                     p2(thmax, Img(w,w,u1.nch)); 

    // setup scan direction
    const int tbase= direction == 1? 4 : 0;
    const int fx   = direction == 1? 0 : u1.nx-1;
    const int fy   = direction == 1? 0 : u1.ny-1;

#pragma omp parallel for shared(p1,p2)
    for (int j=0;j<u1.ny;j++) {
        int y = fy + direction * j;
        for (int i=0;i<u1.nx;i++) {
            int x = fx + direction * i;
            int thid = 0;
            #ifdef _OPENMP // thread id
            thid = omp_get_thread_num();
            #endif
            PointD2 curr(off(x,y,0), off(x,y,1));
            if(curr[0]==0 || curr[1]==0) continue;
            float curr_cost = cost(x,y);

            extract_patch_secure(u1, x, y, p1[thid]);
            
            // scan the neighbors (backward set)   (forward set)
            static const PointD2 neighs[] = {PointD2(0,1),  PointD2(1,0),  PointD2(1,1),   PointD2(-1,1), 
                                           PointD2(0,-1), PointD2(-1,0), PointD2(-1,-1), PointD2(1,-1)};
            for (int trial=0;trial<4; trial++) {
                // position of the neighbor
                PointD2 neigh = PointD2(x,y) + neighs[trial+tbase];
                
                if( !check_inside_image(neigh[0],neigh[1], u1)) continue;

                PointD2 noff( off(neigh[0], neigh[1], 0), off(neigh[0], neigh[1], 1) );

                if(noff[0]==0 || noff[1]==0) continue;
                
                PointD2 tmp = PointD2(x,y) + noff;

                // skip points that fell outside the image or offsets smaller than minoff 
                if( hypot(noff[0],noff[1]) < minoff || 
                    ! check_inside_image(tmp[0],tmp[1], u2)  ) continue;
                
                extract_patch_secure(u2, tmp[0], tmp[1], p2[thid]);

                float new_cost = dist(p1[thid], p2[thid], curr_cost);
                if( new_cost < curr_cost ) {
                    curr_cost=new_cost;
                    curr = noff;
                }
                
            }
            cost(x,y) = curr_cost;
            off(x,y,0) = curr[0];
            off(x,y,1) = curr[1];
        }
    }
}


template<patch_distance_func dist>
void patchmatch(Img &u1, Img &u2, int w, Img &off, Img &cost,int minoff, int maxoff,  int iterations, int randomtrials, bool use_horizontal_off)
{
    RANDOMTRIALS=randomtrials;

    cost.setvalues(INFINITY);
    
    //srand(0);
    for (int i=0;i<iterations;i++)
    {
        printf("iteration %d\n",i);
        // random search
        random_search<dist>(u1, u2, w, off, cost, minoff, maxoff, use_horizontal_off);
      //   // forward propagation
        propagation<dist>(u1, u2, w, off, cost, minoff, maxoff, 1); 
      //   // backward propagation
        propagation<dist>(u1, u2, w, off, cost, minoff, maxoff, -1); 
    }
}

/************ Main Function  **************/

#ifndef DONT_USE_MAIN

int main(int argc, char** argv)
{
    std::string strImageFolder, strPoseFolder, strAltitudeFolder, strGroundRangeFolder, strAnnotationFolder;

    // --- read input data paths --- //
    {
      cxxopts::Options options("data_parsing", "Reads input files...");
      options.add_options()
          ("help", "Print help")
          ("image", "Input folder containing sss image files", cxxopts::value(strImageFolder))
          ("pose", "Input folder containing auv pose files", cxxopts::value(strPoseFolder))
          ("altitude", "Input folder containing auv altitude files", cxxopts::value(strAltitudeFolder))
          ("groundrange", "Input folder containing ground range files", cxxopts::value(strGroundRangeFolder))
          ("annotation", "Input folder containing annotation files", cxxopts::value(strAnnotationFolder));

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
    }

    // --- parse input data --- //
    std::vector<cv::Mat> vmImgs;
    std::vector<cv::Mat> vmPoses;
    std::vector<std::vector<double>> vvAltts;
    std::vector<std::vector<double>> vvGranges;
    std::vector<cv::Mat> vmAnnos;
    Util::LoadInputData(strImageFolder,strPoseFolder,strAltitudeFolder,strGroundRangeFolder,strAnnotationFolder,
                        vmImgs,vmPoses,vvAltts,vvGranges,vmAnnos);

    // --- construct frame --- //
    int test_num = vmImgs.size();
   //  int test_num = 2;
    std::vector<Frame> test_frames;
    for (size_t i = 0; i < test_num; i++)
        test_frames.push_back(Frame(i,vmImgs[i],vmPoses[i],vvAltts[i],vvGranges[i],vmAnnos[i]));

    // --- find correspondences between each pair of frames --- //
    for (size_t i = 0; i < test_frames.size(); i++)
    {
        cv::Mat new_anno_kps = test_frames[i].anno_kps.;
        
        for (size_t j = 0; j < test_frames.size(); j++)
        {
            if (i==j)
              continue;
            
            int count = 0, count2 = 0;

            cv::Mat img_s = test_frames[i].norm_img;
            cv::Mat img_t;
            if (abs(test_frames[i].img_id-test_frames[j].img_id)%2!=0)
                flip(test_frames[j].norm_img,img_t,-1);
            else
                img_t = test_frames[j].norm_img;

            // get initial flow from Dead-reckoning Prior
            std::vector<cv::Mat> ini_flow = FEAmatcher::IniFlow(test_frames[i],test_frames[j]);

            struct Img odisp(test_frames[i].norm_img.cols, test_frames[i].norm_img.rows, 2);
            struct Img odisp_ini(test_frames[i].norm_img.cols, test_frames[i].norm_img.rows, 2); // for comparison
            for (size_t k = 0; k < ini_flow.size(); k++)
                for (size_t m = 0; m < ini_flow[k].rows; m++)
                        for (size_t n = 0; n < ini_flow[k].cols; n++)
                        {  
                        odisp[count] = ini_flow[k].at<float>(m,n);
                        odisp_ini[count] = ini_flow[k].at<float>(m,n);                  
                        count++;
                        }

            cout << "initial flow generation completed..." << endl;          


            struct Img test1(img_s.cols, img_s.rows);
            count = 0;
            for (size_t m = 0; m < img_s.rows; m++)
            {
                for (size_t n = 0; n < img_s.cols; n++)
                {
                    test1[count] = (float)img_s.at<uchar>(m,n);
                    count++;
                }    
            }
            
            struct Img test2(img_t.cols, img_t.rows);
            count = 0;
            for (size_t m = 0; m < img_t.rows; m++)
            {
                for (size_t n = 0; n < img_t.cols; n++)
                {
                    test2[count] = (float)img_t.at<uchar>(m,n);
                    count++;
                }    
            }

            // --- (Img &u1, Img &u2, int w, Img &off, Img &cost,int minoff, int maxoff,  int iterations, int randomtrials, bool use_horizontal_off)
            struct Img ocost(img_s.cols, img_s.rows);
            patchmatch<distance_patch_NCC>(test1, test2, 13, odisp, ocost, 0, 8, 10, 10, false);

            // --- demonstrate --- //
            std::vector<cv::DMatch> TemperalMatches;
            std::vector<cv::KeyPoint> PreKeys, CurKeys;
            count = 0;
            // for (size_t n = 0; n < test_frames[i].anno_kps.rows; n=n+20)
            // {
            //    const int r = test_frames[i].anno_kps.at<int>(n,2);
            //    const int c = test_frames[i].anno_kps.at<int>(n,3);
            //    const double c_offset = odisp(c,r,0);
            //    const double r_offset = odisp(c,r,1);
            //    if (r_offset==0 || c_offset==0)
            //        continue;   
            //    PreKeys.push_back(cv::KeyPoint(c,r,0,0,0,-1));               
            //    CurKeys.push_back(cv::KeyPoint(c+c_offset,r+r_offset,0,0,0,-1));
            //    TemperalMatches.push_back(cv::DMatch(count,count,0));
            //    count++;
            //    // cout << r << " " << c << " " << r+r_offset << " " << c+c_offset << endl;
            // }

            ofstream save_rc_ini, save_rc_fnl;
            string path1 = "../rc_ini.txt";
            save_rc_ini.open(path1.c_str(),ios::trunc);
            string path2 = "../rc_fnl.txt";
            save_rc_fnl.open(path2.c_str(),ios::trunc);

            // --- metric error --- //
            float e_r = 0, e_c = 0, e_rc = 0;
            int test_num = 0;
            for (size_t n = 0; n < test_frames[i].anno_kps.rows; n=n+1)
            {
                const int r = test_frames[i].anno_kps.at<int>(n,2);
                const int c = test_frames[i].anno_kps.at<int>(n,3);

                const double c_offset = odisp(c,r,0);
                const double r_offset = odisp(c,r,1);
                const double c_offset_ini = odisp_ini(c,r,0);
                const double r_offset_ini = odisp_ini(c,r,1);
                const double min_cost = ocost(c,r);

                if (r_offset==0 || c_offset==0)
                    continue;

                int r_gt = 0, c_gt = 0;
                if (test_frames[i].anno_kps.at<int>(n,1)==test_frames[j].img_id)
                {
                    if (abs(test_frames[i].img_id-test_frames[j].img_id)%2!=0)
                    {
                        r_gt = test_frames[j].norm_img.rows-test_frames[i].anno_kps.at<int>(n,4)-1;
                        c_gt = test_frames[j].norm_img.cols-test_frames[i].anno_kps.at<int>(n,5)-1;
                    }
                    else
                    {
                        r_gt = test_frames[i].anno_kps.at<int>(n,4);
                        c_gt = test_frames[i].anno_kps.at<int>(n,5);
                    }
                    test_num++;
                }
                else
                    continue;

                const float e_r_tmp = abs(r_gt - (r+r_offset));
                const float e_c_tmp = abs(c_gt - (c+c_offset));
                const float e_rc_tmp = (r_gt - (r+r_offset))*(r_gt - (r+r_offset)) + (c_gt - (c+c_offset))*(c_gt - (c+c_offset));
                const float e_rc_tmp_ini = (r_gt - (r+r_offset_ini))*(r_gt - (r+r_offset_ini)) + (c_gt - (c+c_offset_ini))*(c_gt - (c+c_offset_ini));

                save_rc_ini << sqrt(e_rc_tmp_ini) << endl;
                save_rc_fnl << sqrt(e_rc_tmp) << endl;
                
                // if (!(sqrt(e_rc_tmp)<=1.2*sqrt(e_rc_tmp_ini) || (sqrt(e_rc_tmp)<6) || (e_r_tmp<=-1 && e_c_tmp<=-1)))
                if (sqrt(e_rc_tmp)>30 && sqrt(e_rc_tmp)<1000)
                {
                    cout << "error per point (r and c): " << abs(r_gt - (r+r_offset_ini)) << "->" << abs(r_gt - (r+r_offset)) << " ";
                    cout << abs(c_gt - (c+c_offset_ini)) << "->" << abs(c_gt - (c+c_offset)) << " " << min_cost << endl;

                    PreKeys.push_back(cv::KeyPoint(c,r,0,0,0,-1));               
                    // CurKeys.push_back(cv::KeyPoint(c+c_offset_ini,r+r_offset_ini,0,0,0,-1));
                    CurKeys.push_back(cv::KeyPoint(c+c_offset,r+r_offset,0,0,0,-1));
                    // CurKeys.push_back(cv::KeyPoint(c_gt,r_gt,0,0,0,-1));
                    TemperalMatches.push_back(cv::DMatch(count,count,0));
                    count++;

                    // cv::Mat img_matches;
                    // cv::drawMatches(img_s, PreKeys, img_t, CurKeys, TemperalMatches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                    //                vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                    // cv::namedWindow("temperal matches", cv::WINDOW_NORMAL);
                    // cv::imshow("temperal matches", img_matches);
                    // cv::waitKey(0);
                }
                // cout << "error per point (r and c): " << abs(r_gt - (r+r_offset)) << " " << abs(c_gt - (c+c_offset)) << endl;
                e_r = e_r + e_r_tmp;
                e_c = e_c + e_c_tmp;
                e_rc = e_rc + e_rc_tmp;    

            }
            e_r = e_r/test_num;
            e_c = e_c/test_num;
            e_rc = sqrt(e_rc/test_num);
            cout << "Metric Error btw " << test_frames[i].img_id << " and " << test_frames[j].img_id << " (r c rc %): ";
            cout << e_r << " " << e_c << " " << e_rc << " " << (float)count/test_num*100 << "%" << " (" << count << "/" << test_num << ")"  << endl;

            
            cv::Mat img_matches;
            cv::drawMatches(img_s, PreKeys, img_t, CurKeys, TemperalMatches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                            vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            cv::namedWindow("temperal matches", cv::WINDOW_NORMAL);
            cv::imshow("temperal matches", img_matches);
            cv::waitKey(0);

        }
    }

    // // --- find correspondences between each pair of frames --- //
    // for (size_t i = 0; i < test_frames.size(); i++)
    // {
    //     cv::Mat new_anno_kps = test_frames[i].anno_kps.;

    //     for (size_t j = i+1; j < test_frames.size(); j++)
    //       {
    //         int count = 0, count2 = 0;

    //         cv::Mat img_s = test_frames[i].norm_img;
    //         cv::Mat img_t;
    //         if (abs(test_frames[i].img_id-test_frames[j].img_id)%2!=0)
    //             flip(test_frames[j].norm_img,img_t,-1);
    //         else
    //             img_t = test_frames[j].norm_img;

    //         // get initial flow from Dead-reckoning Prior
    //         std::vector<cv::Mat> ini_flow = FEAmatcher::IniFlow(test_frames[i],test_frames[j]);

    //         struct Img odisp(test_frames[i].norm_img.cols, test_frames[i].norm_img.rows, 2);
    //         struct Img odisp_ini(test_frames[i].norm_img.cols, test_frames[i].norm_img.rows, 2); // for comparison
    //         for (size_t k = 0; k < ini_flow.size(); k++)
    //            for (size_t m = 0; m < ini_flow[k].rows; m++)
    //                  for (size_t n = 0; n < ini_flow[k].cols; n++)
    //                  {  
    //                     odisp[count] = ini_flow[k].at<float>(m,n);
    //                     odisp_ini[count] = ini_flow[k].at<float>(m,n);                  
    //                     count++;
    //                  }

    //         cout << "initial flow generation completed..." << endl;          

    //         // struct Img odisp(img_s.cols, img_s.rows, 2);
    //         // odisp.setvalues(0);

    //         struct Img test1(img_s.cols, img_s.rows);
    //         count = 0;
    //         for (size_t m = 0; m < img_s.rows; m++)
    //         {
    //            for (size_t n = 0; n < img_s.cols; n++)
    //            {
    //               test1[count] = (float)img_s.at<uchar>(m,n);
    //               count++;
    //            }    
    //         }
            
    //         struct Img test2(img_t.cols, img_t.rows);
    //         count = 0;
    //         for (size_t m = 0; m < img_t.rows; m++)
    //         {
    //            for (size_t n = 0; n < img_t.cols; n++)
    //            {
    //               test2[count] = (float)img_t.at<uchar>(m,n);
    //               count++;
    //            }    
    //         }

    //         // (Img &u1, Img &u2, int w, Img &off, Img &cost,int minoff, int maxoff,  int iterations, int randomtrials, bool use_horizontal_off)
    //         struct Img ocost(img_s.cols, img_s.rows);
    //         patchmatch<distance_patch_NCC>(test1, test2, 13, odisp, ocost, 0, 8, 10, 10, false);

    //         // cv::Mat check_mask(test_frames[i].norm_img.size(), CV_8UC1, Scalar(255));
    //         // for (int y=0;y<odisp.ny;y++) {
    //         //    for (int x=0;x<odisp.nx;x++) {

    //         //        if (odisp(x,y,0)!=0 || odisp(x,y,1)!=0)
    //         //        {
    //         //          check_mask.at<bool>(y,x) = 0;
    //         //        }
                                     
    //         //    }
    //         // }
    //         // cv::Mat out_demo;
    //         // check_mask.copyTo(out_demo);
    //         // cv::namedWindow("filtered mask", cv::WINDOW_AUTOSIZE);
    //         // cv::imshow("filtered mask", out_demo);
    //         // cv::waitKey(0);

    //         // // generate the backprojected image
    //         // struct Img syn = Img(test1.nx, test1.ny, test1.nch);
    //         // for(int x=0;x<test1.nx;x++)
    //         //    for(int y=0;y<test1.ny;y++){
    //         //       PointD2 q = PointD2(odisp(x,y,0),odisp(x,y,1));
    //         //       for(int c=0;c<test1.nch;c++)
    //         //          if( check_inside_image(x+q[0], y+q[1], test2) ) 
    //         //             syn(x,y,c) = test2(x+q[0],y+q[1],c);
    //         //          else 
    //         //             // syn(x,y,c) = 0;
    //         //             syn(x,y,c) = test1(x,y,c);
    //         //    }
            
    //         // cout << "saving files..." << endl;
    //         // iio_write_vector_split("offset_new.tif", odisp);
    //         // iio_write_vector_split("backproj_new.png", syn);

    //         // --- demonstrate --- //
    //         std::vector<cv::DMatch> TemperalMatches;
    //         std::vector<cv::KeyPoint> PreKeys, CurKeys;
    //         count = 0;
    //         // for (size_t n = 0; n < test_frames[i].anno_kps.rows; n=n+20)
    //         // {
    //         //    const int r = test_frames[i].anno_kps.at<int>(n,2);
    //         //    const int c = test_frames[i].anno_kps.at<int>(n,3);
    //         //    const double c_offset = odisp(c,r,0);
    //         //    const double r_offset = odisp(c,r,1);
    //         //    if (r_offset==0 || c_offset==0)
    //         //        continue;   
    //         //    PreKeys.push_back(cv::KeyPoint(c,r,0,0,0,-1));               
    //         //    CurKeys.push_back(cv::KeyPoint(c+c_offset,r+r_offset,0,0,0,-1));
    //         //    TemperalMatches.push_back(cv::DMatch(count,count,0));
    //         //    count++;
    //         //    // cout << r << " " << c << " " << r+r_offset << " " << c+c_offset << endl;
    //         // }

    //         ofstream save_rc_ini, save_rc_fnl;
    //         string path1 = "../rc_ini.txt";
    //         save_rc_ini.open(path1.c_str(),ios::trunc);
    //         string path2 = "../rc_fnl.txt";
    //         save_rc_fnl.open(path2.c_str(),ios::trunc);

    //         // --- metric error --- //
    //         float e_r = 0, e_c = 0, e_rc = 0;
    //         int test_num = 0;
    //         for (size_t n = 0; n < test_frames[i].anno_kps.rows; n=n+1)
    //         {
    //            const int r = test_frames[i].anno_kps.at<int>(n,2);
    //            const int c = test_frames[i].anno_kps.at<int>(n,3);

    //            const double c_offset = odisp(c,r,0);
    //            const double r_offset = odisp(c,r,1);
    //            const double c_offset_ini = odisp_ini(c,r,0);
    //            const double r_offset_ini = odisp_ini(c,r,1);
    //            const double min_cost = ocost(c,r);

    //            if (r_offset==0 || c_offset==0)
    //                continue;

    //            int r_gt = 0, c_gt = 0;
    //            if (test_frames[i].anno_kps.at<int>(n,1)==test_frames[j].img_id)
    //            {
    //               if (abs(test_frames[i].img_id-test_frames[j].img_id)%2!=0)
    //               {
    //                  r_gt = test_frames[j].norm_img.rows-test_frames[i].anno_kps.at<int>(n,4)-1;
    //                  c_gt = test_frames[j].norm_img.cols-test_frames[i].anno_kps.at<int>(n,5)-1;
    //               }
    //               else
    //               {
    //                  r_gt = test_frames[i].anno_kps.at<int>(n,4);
    //                  c_gt = test_frames[i].anno_kps.at<int>(n,5);
    //               }
    //               test_num++;
    //            }
    //            else
    //              continue;

    //            const float e_r_tmp = abs(r_gt - (r+r_offset));
    //            const float e_c_tmp = abs(c_gt - (c+c_offset));
    //            const float e_rc_tmp = (r_gt - (r+r_offset))*(r_gt - (r+r_offset)) + (c_gt - (c+c_offset))*(c_gt - (c+c_offset));
    //            const float e_rc_tmp_ini = (r_gt - (r+r_offset_ini))*(r_gt - (r+r_offset_ini)) + (c_gt - (c+c_offset_ini))*(c_gt - (c+c_offset_ini));

    //            save_rc_ini << sqrt(e_rc_tmp_ini) << endl;
    //            save_rc_fnl << sqrt(e_rc_tmp) << endl;
               
    //            // if (!(sqrt(e_rc_tmp)<=1.2*sqrt(e_rc_tmp_ini) || (sqrt(e_rc_tmp)<6) || (e_r_tmp<=-1 && e_c_tmp<=-1)))
    //            if (sqrt(e_rc_tmp)>30 && sqrt(e_rc_tmp)<1000)
    //            {
    //               cout << "error per point (r and c): " << abs(r_gt - (r+r_offset_ini)) << "->" << abs(r_gt - (r+r_offset)) << " ";
    //               cout << abs(c_gt - (c+c_offset_ini)) << "->" << abs(c_gt - (c+c_offset)) << " " << min_cost << endl;

    //               PreKeys.push_back(cv::KeyPoint(c,r,0,0,0,-1));               
    //               // CurKeys.push_back(cv::KeyPoint(c+c_offset_ini,r+r_offset_ini,0,0,0,-1));
    //               CurKeys.push_back(cv::KeyPoint(c+c_offset,r+r_offset,0,0,0,-1));
    //               // CurKeys.push_back(cv::KeyPoint(c_gt,r_gt,0,0,0,-1));
    //               TemperalMatches.push_back(cv::DMatch(count,count,0));
    //               count++;

    //               // cv::Mat img_matches;
    //               // cv::drawMatches(img_s, PreKeys, img_t, CurKeys, TemperalMatches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
    //               //                vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //               // cv::namedWindow("temperal matches", cv::WINDOW_NORMAL);
    //               // cv::imshow("temperal matches", img_matches);
    //               // cv::waitKey(0);
    //            }
    //            // cout << "error per point (r and c): " << abs(r_gt - (r+r_offset)) << " " << abs(c_gt - (c+c_offset)) << endl;
    //            e_r = e_r + e_r_tmp;
    //            e_c = e_c + e_c_tmp;
    //            e_rc = e_rc + e_rc_tmp;    

    //         }
    //         e_r = e_r/test_num;
    //         e_c = e_c/test_num;
    //         e_rc = sqrt(e_rc/test_num);
    //         cout << "Metric Error btw " << test_frames[i].img_id << " and " << test_frames[j].img_id << " (r c rc %): ";
    //         cout << e_r << " " << e_c << " " << e_rc << " " << (float)count/test_num*100 << "%" << " (" << count << "/" << test_num << ")"  << endl;

            
    //         cv::Mat img_matches;
    //         cv::drawMatches(img_s, PreKeys, img_t, CurKeys, TemperalMatches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
    //                        vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //         cv::namedWindow("temperal matches", cv::WINDOW_NORMAL);
    //         cv::imshow("temperal matches", img_matches);
    //         cv::waitKey(0);

    //       }

    // // --- optimize trajectory between images --- //
    // // Optimizer::TrajOptimizationPair(test_frames[0], test_frames[2]);
    // //  Optimizer::TrajOptimizationAll(test_frames);
    
    // }



    return 0;
}

// int main(int argc, char* argv[]) 
// {

// 	//read the parameters
//    char *in_disp_file = pick_option(&argc, &argv, (char*) "d", (char*) "");
//    int dmin = atoi(pick_option(&argc, &argv, (char*) "r", (char*) "0"));
//    int dmax = atoi(pick_option(&argc, &argv, (char*) "R", (char*) "30"));
//    char* method = pick_option(&argc, &argv, (char*) "t", (char*) "SSD");   //{census|ad|sd|ncc|btad|btsd}
//    int iterations = atoi(pick_option(&argc, &argv, (char*) "i", (char*) "5"));
//    int w = atoi(pick_option(&argc, &argv, (char*) "w", (char*) "7"));
//    bool use_horizontal_off = pick_option(&argc, &argv, (char*) "h", NULL);

//    // std::cout << method << std::endl;

// 	/* patameter parsing - parameters*/
// 	if(argc<4)
// 	{
// 		fprintf (stderr, "too few parameters\n");
// 		fprintf (stderr, "   usage: %s  [-i iter(5)] [-w window(7)] [-r min_offset(0)] [-R max_offset(10)] [-h] [-d init_disp] u v out [cost [backflow]]\n",argv[0]);
// 		fprintf (stderr, "        [-t  distance(ad)]: distance = {SSD|SAD|ZSSD|ZSAD|NCC} \n");
// 		return 1;
// 	}
	
// 	int i = 1;
// 	char* f_u     = (argc>i) ? argv[i] : NULL;      i++;
// 	char* f_v     = (argc>i) ? argv[i] : NULL;      i++;
// 	char* f_out   = (argc>i) ? argv[i] : NULL;      i++;
// 	char* f_cost  = (argc>i) ? argv[i] : NULL;      i++;
// 	char* f_back  = (argc>i) ? argv[i] : NULL;      i++;
	
// 	printf("%d %d\n", dmin, dmax);
	
	
// 	// read input
// 	struct Img u = iio_read_vector_split(f_u);
// 	struct Img v = iio_read_vector_split(f_v);

//    //remove_nonfinite_values_Img(u, 0);
//    //remove_nonfinite_values_Img(v, 0);

// 	// call
// 	struct Img ocost(u.nx, u.ny);
//    struct Img odisp(u.nx, u.ny, 2);
//    odisp.setvalues(dmin);

//    if(strcmp (in_disp_file,"")!=0 ){
//    	odisp = iio_read_vector_split(in_disp_file);
//    }


//    if (strcmp (method,"SSD")==0)
//       patchmatch<distance_patch_SSD>(u, v, w, odisp, ocost,dmin, dmax,  iterations, 5, use_horizontal_off);
//    else if (strcmp (method,"SAD")==0)
//       patchmatch<distance_patch_SAD>(u, v, w, odisp, ocost,dmin, dmax,  iterations, 5, use_horizontal_off);
//    else if (strcmp (method,"ZSSD")==0)
//       patchmatch<distance_patch_ZSSD>(u, v, w, odisp, ocost,dmin, dmax,  iterations, 5, use_horizontal_off);
//    else if (strcmp (method,"ZSAD")==0)
//       patchmatch<distance_patch_ZSAD>(u, v, w, odisp, ocost,dmin, dmax,  iterations, 5, use_horizontal_off);
//    else if (strcmp (method,"NCC")==0)
//       patchmatch<distance_patch_NCC>(u, v, w, odisp, ocost,dmin, dmax,  iterations, 5, use_horizontal_off);

// 	// save the disparity
	
// 	// generate the backprojected image
// 	struct Img syn = Img(u.nx, u.ny, u.nch);
// 	for(int x=0;x<u.nx;x++)
// 		for(int y=0;y<u.ny;y++){
// 			PointD2 q = PointD2(odisp(x,y,0),odisp(x,y,1));
// 			for(int c=0;c<u.nch;c++)
// 				if( check_inside_image(x+q[0], y+q[1], v) ) 
// 					syn(x,y,c) = v(x+q[0],y+q[1],c);
// 				else 
//                syn(x,y,c) = 0;
// 					// syn(x,y,c) = u(x,y,c);
// 		}
	
	// iio_write_vector_split(f_out, odisp);
// 	if(f_cost) iio_write_vector_split(f_cost, ocost);
// 	if(f_back) iio_write_vector_split(f_back, syn);
	
// 	return 0;
// }

#endif
