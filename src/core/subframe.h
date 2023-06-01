
#ifndef SUBFRAME_H
#define SUBFRAME_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace Diasss
{

    class SubFrame
    {

    public:

        // Constructor 
        SubFrame(){};
        
        int subframe_id;
        int start_ping, centre_ping, end_ping;

        std::vector<int> corres_ids; // the position id of correspondences (of this subframe) in the full frame;

        std::vector<std::vector<int>> asso_sf_corres_ids; // the associated subframe correspondences' ids (subset of the 'corres_ids' above);
        std::vector<std::pair<int,int>> asso_sf_ids; // the associated parent frame and subframe ids;


    private:



    };

}

#endif