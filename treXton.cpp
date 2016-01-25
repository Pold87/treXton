#include <iostream>
#include <string>

// Include OpenCV libraries

#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

#include <random>


using namespace std;
using namespace cv;

// Currently works for 1D only
Mat extract_one_texton(const Mat &img,
                   const int &x,
                   const int &y,
                   const int &texton_size_w,
                   const int &texton_size_h) {

   Mat texton;
   texton = cv::Mat(img, cv::Rect(x, y, texton_size_w, texton_size_h)).clone();

   cout << texton << endl;

   cv::Mat texton_flat;
   texton_flat = texton.reshape(1, 1);

   cout << texton_flat << endl;

}


int main(int argc, char *argv[])
{

   // Read in image
   Mat img;
   img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

   // Check for invalid input
   if (!img.data) {
      std::cout << "Could not open or find the image" << std::endl;
      return -1;
   }

   //imshow("frame", img);

   extract_one_texton(img, 0, 0, 5, 5);

   waitKey(0);

   return 0;
}
