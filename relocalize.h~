#ifndef RELOCALIZE_H
#define RELOCALIZE_H

#include <string>

// Include OpenCV libraries
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include <boost/python.hpp>


class Relocalizer {

  std::string ref_img_path; // Path of the reference image
  cv::Mat ref_img; // Reference image as matrix
  cv::Mat ref_img_c; // Reference image as matrix (color)
  std::vector<cv::KeyPoint> kp_ref; // Keypoints of the ref image
  cv::Mat des_ref; // Descriptors of the keypoints of the ref image
  cv::Ptr<cv::xfeatures2d::SURF> detector; // Feature detector
  cv::BFMatcher matcher;
      
   std::vector<cv::KeyPoint> matched_query, matched_map, inliers_query, inliers_ref;
   std::vector<cv::DMatch> good_matches;

 public:

   Relocalizer(std::string ref_img_path);
   std::vector<float> calcLocation(cv::Mat query_img);
   boost::python::list calcLocationFromPath(std::string query_img_path);
  
};



using namespace boost::python;

BOOST_PYTHON_MODULE(relocalize)
{
  class_<Relocalizer>("Relocalizer", init<std::string>())
    .def("calcLocation", &Relocalizer::calcLocation)
    .def("calcLocationFromPath", &Relocalizer::calcLocationFromPath)
    ;
};

#endif
