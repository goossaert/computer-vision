// (c)2010 - Emmanuel Goossaert
// Under GNU License 3.0
#include <iostream>
#include <vector>
using std::vector;

#include "cv.h"
#include "highgui.h"

#include "had.h"


int main(int argc, char** argv)
{
    if( argc < 4 )
    {
        std::cout << "usage: " << argv[0] << " " << "detection_rate out_segmentation.jpg test_image.jpg training_image01.jpg training_image02.jpg ..." << std::endl;
        exit( 0 );
    }

    float detection_rate = atof( argv[ 1 ] );
    std::cout << "Detection rate: " << detection_rate << std::endl;

    cv::Mat image_test = cv::imread( argv[ 3 ] );
    std::cout << "Test image: " << argv[ 3 ] << std::endl;

    vector<cv::Mat> images_training;
    for( int i = 4; i < argc; ++i )
    {
        std::cout << "Training image: " << argv[ i ] << std::endl;
        images_training.push_back( cv::imread( argv[ i ] ) );
    }

    had::MultipleLCM lcm( images_training, detection_rate );

    cv::Mat classification, image_classification;
    lcm.classify( image_test, classification );
    lcm.classificationToImage( classification, image_classification );
    cv::imwrite( argv[ 2 ], image_classification );

    std::cout << "Blue: foreground, Green: background, Red: shadow, Black: highlight" << std::endl;

    cv::namedWindow( "Classification", 0 );
    cv::imshow( "Classification", image_classification );
    cv::waitKey();

    return 0;
}
