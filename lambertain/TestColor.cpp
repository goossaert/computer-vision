#include <iostream>
#include <vector>
using std::vector;

#include "cv.h"
#include "highgui.h"

#include "had.h"


int main(int argc, char** argv)
{
    if( argc < 9 )
    {
        std::cout << "usage: " << argv[0] << " " << "detection_rate out_segmentation.jpg out_region.jpg  image.jpg x y width height" << std::endl;
        exit( 0 );
    }

    float detection_rate = atof( argv[ 1 ] );
    std::cout << "Detection rate: " << detection_rate << std::endl;

    cv::Mat image = cv::imread( argv[ 4 ] );
    std::cout << "Image: " << argv[ 4 ] << std::endl;

    vector<cv::Rect> rectangles;
    rectangles.push_back( cv::Rect( atoi(argv[5]),
                                    atoi(argv[6]),
                                    atoi(argv[7]),
                                    atoi(argv[8])
                                  )
                        );

    cv::Mat image_rect = image.clone();
    cv::rectangle( image_rect, rectangles[0], cv::Scalar( 255 ) );
    cv::namedWindow( "image", 0 );
    cv::imshow( "image", image_rect );
    cv::imwrite( argv[ 3 ], image_rect );

    had::SingleLCM lcm( image, detection_rate, rectangles );

    cv::Mat classification, image_classification;
    lcm.classify( image, classification );
    lcm.classificationToImage( classification, image_classification );
    cv::imwrite( argv[ 2 ], image_classification );

    std::cout << "Blue: foreground, Green: background, Red: shadow, Black: highlight" << std::endl;

    cv::namedWindow( "Classification", 0 );
    cv::imshow( "Classification", image_classification );
    cv::waitKey();

    return 0;
}
