// (c)2010 - Emmanuel Goossaert
// Under GNU License 3.0
// This code was based on pre-existing sample code that I got from:
// http://opencv.willowgarage.com/documentation/cpp/reading_and_writing_images_and_video.html
#include <iostream>
#include <sstream>
using std::stringstream;

#include "cv.h"
#include "highgui.h"

using namespace cv;


int main(int, char**)
{
    VideoCapture cap(0); // open the default camera
    if( !cap.isOpened() )  // check if we succeeded
        return -1;

    std::cout << std::endl
              << "Press ESCAPE to exit, or any key to save a frame to a file..." << std::endl;

    int i = 0;
    Mat frame;
    while( true )
    {
        cap >> frame; // get a new frame from camera
        imshow("frames", frame);

        int key = waitKey( 30 );
        if( key == 27 ) // escape code
        {
            break; // break instead of exiting, so that the camera can be deinitialized properly
        }
        else if( key >= 0 )
        {
            stringstream ss;
            ss << "frame" << i << ".jpg";
            imwrite( ss.str(), frame );
            std::cout << "Frame saved to \"" << ss.str() << "\"" << std::endl;
            ++i;
        }
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
