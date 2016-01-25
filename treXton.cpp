#include <iostream>
#include <string>

// Include OpenCV libraries

#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

#include <random>
#include <time.h>
#include <stdlib.h>


using namespace std;
using namespace cv;

// Currently works for 1D only
int[] extract_one_texton(const Mat &img,
                   const int &x,
                   const int &y,
                   const int &texton_size_w,
                   const int &texton_size_h) {

   Mat texton;
   texton = cv::Mat(img, cv::Rect(x, y, texton_size_w, texton_size_h)).clone();

   // Print texton
   cout << texton << endl;

   //cv::Mat texton_flat;
   vector<int> texton_flat;
   texton_flat = texton.reshape(1, 1);

   // Print flattend texton
   //cout << texton_flat << endl;

   int texton_flat[texton_size_w * texton_size_h];

   return texton_flat;

}


int label_patch(vector< vector<int> > texton_dict){



}


int main(int argc, char *argv[])
{

  int amount_textons = 200;
  int texton_size = 5;

   // Read in image
   Mat img;
   img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

   // Check for invalid input
   if (!img.data) {
      std::cout << "Could not open or find the image" << std::endl;
      return -1;
   }

   //imshow("frame", img);

   vector<int> texton;
   vector< vector<int> > textons;

   // Create seed for random number
   srand(time(NULL));

   // Extract image patches (amount is specified by 'amount_textons')
   for (int i = 0; i < amount_textons; ++i) {
     int x = (rand() % (img.cols - 6));
     int y = (rand() % (img.rows - 6));
     texton = extract_one_texton(img, x, y, 
				   texton_size, texton_size);
     textons.push_back(texton);
   }

   waitKey(0);

   return 0;
}
