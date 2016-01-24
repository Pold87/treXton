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



int main(int argc, char *argv[])
{
  Mat img;
  img = imread(argv[1], CV_LOAD_IMAGE_COLOR);

  return 0;
}
