// No copyright - Volker

#include <stdio.h>

// Include standard libraries
#include <iostream>
#include <string>
#include <typeinfo>
#include <fstream>
#include <chrono>

// Include OpenCV libraries

#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

// Math libraries
#include <math.h>
#include <cmath>

using namespace std;
using namespace cv;


// Suited for calling from python
boost::python::list myfastdot() {

  // Read in query image
  cv::Mat query_img = cv::imread(query_img_path);

  cv::Point2f loc;
  loc = calcLocation(query_img);

  boost::python::list python_list;
  python_list.append(loc.x);
  python_list.append(loc.y);

  return python_list;

}
