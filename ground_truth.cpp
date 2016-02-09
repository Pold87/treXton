#include <iostream>
#include <string>
#include "relocalize.h"

// Include OpenCV libraries
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"


using namespace std;

int main(int argc, char *argv[])
{

  if (argc == 1) {
    
    cout << "***\nUsage: ./ground_truth <num_pics>\n***" << std::endl;
 
    return -1;
  }

  // int num_pics = argv[1];
  string basedir = "../image_recorder/playing_mat";
  int num_pics = std::atoi(argv[1]);

  // Read in background image
  string background_map = "../draug/img/bestnewmat.png";
  cv::Mat background_img = cv::imread(background_map);
  int img_width = background_img.cols;
  
  /* Open file for saving estimates */
  FILE* fp;
  fp = fopen("sift_targets.csv", "w");
  fprintf(fp, "id,x,y,matches"); /* Print header */

  // Construct relocalizer with reference image path
  Relocalizer relocalizer(background_map);

  int i;
  for (i = 0; i < num_pics; i++) {
    cout << i << "\n";
    /* Read image */
    string img_path = basedir + "/" + to_string(i) + ".png";
    cout << img_path << "\n";
    cv::Mat img = cv::imread(img_path);

    // Get estimation (x, y) in pixels from relocalizer
    std::vector<float> coords = relocalizer.calcLocation(img);
    fprintf(fp, "\n%d,%f,%f,%d", i, coords[0], img_width - coords[1],(int) coords[2]);
  }

  fclose(fp);
  
  return 0;
}
