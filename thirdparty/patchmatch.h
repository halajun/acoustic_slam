#ifndef PATCHMATCH_H_
#define PATCHMATCH_H_

#include "stdlib.h"
#include <math.h>

#include "point.h"
#ifdef _OPENMP 
#include <omp.h>
#endif //_OPENMP

//// a structure to wrap images 
#include "img.h"

namespace Diasss
{

using namespace std;

int RANDOMTRIALS=10;

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

//signature of all patch distance functions
typedef float(*patch_distance_func)(const Img &, const Img &, const float&); 

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
        // printf("iteration %d\n",i);
        // random search
        random_search<dist>(u1, u2, w, off, cost, minoff, maxoff, use_horizontal_off);
      //   // forward propagation
        propagation<dist>(u1, u2, w, off, cost, minoff, maxoff, 1); 
      //   // backward propagation
        propagation<dist>(u1, u2, w, off, cost, minoff, maxoff, -1); 
    }
}


}

#endif /* PATCHMATCH_H_ */