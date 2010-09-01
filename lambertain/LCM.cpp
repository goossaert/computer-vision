// (c)2010 - Emmanuel Goossaert
// Under GNU License 3.0
#include "LCM.hpp"

void had::LCM::showImage( const string name, const cv::Mat& image, int col, int row )
{
    // The sizes of 60 and 200 are magic values that I have choosen based upon
    // my own system. Change them to better fit your screen and system.
    int size_captionbar = 60;
    int size_window = 200;
    int x = col * size_window;
    int y = row * size_window;
    int width = size_window;
    int height = size_window;

    // Need to add this due to a bug in OpenCV 2.1
    if( row <= 3 ) y += row * size_captionbar;

    cv::namedWindow( name.c_str(), 0 );

    cvResizeWindow( name.c_str(), width, height );
    cvMoveWindow( name.c_str(), x, y );

    cv::imshow( name.c_str(), image );
}


void had::LCM::fillRectangle( cv::Mat& io_image,
                                       const cv::Rect& rect,
                                       const cv::Scalar value )
{
    cv::rectangle( io_image,
                   cv::Point( rect.x, rect.y ),
                   cv::Point( rect.x + rect.width, rect.y + rect.height ),
                   value,
                   CV_FILLED );
}


void had::LCM::fillRectangles( cv::Mat& io_image,
                                        const vector<cv::Rect>& rectangles,
                                        const cv::Scalar value )
{
    for( vector<cv::Rect>::const_iterator it = rectangles.begin(); it != rectangles.end(); ++it )
    {
        fillRectangle( io_image, *it, value );
    }
}


float had::LCM::computeBrightnessDenominator( const cv::Scalar& mean,
                                              const cv::Scalar& stddev )
{
    // See Horprasert et al., 1999, Eq. 5
    // See Yacoob and Davis, 2006, Eq. 3
    float b = mean[ 0 ] / stddev[ 0 ];
    float g = mean[ 1 ] / stddev[ 1 ];
    float r = mean[ 2 ] / stddev[ 2 ];
    float denom = b * b + g * g + r * r;
    if( denom == 0 ) denom = 1;
    return denom;
}


float had::LCM::computeBrightnessDistortion( const cv::Vec3b& pixel,
                                             const cv::Scalar& brightness )
{
    // See Horprasert et al., 1999, Eq. 5
    // See Yacoob and Davis, 2006, Eq. 3
    
    // There is possibly a typo in Yacoob and Davis, that I have fixed.
    // In order to get the formula as they presented it, just square
    // every pixel value below.
    return   (float) pixel[ 0 ] * brightness[ 0 ]
           + (float) pixel[ 1 ] * brightness[ 1 ]
           + (float) pixel[ 2 ] * brightness[ 2 ];
}


float had::LCM::computeChromacityDistortion( const cv::Vec3b& pixel,
                                             const cv::Scalar& mean,
                                             const cv::Scalar& stddev,
                                                        float bdist )
{
    // See Horprasert et al., 1999, Eq. 6
    // See Yacoob and Davis, 2006, Eq. 4
    // _stddev is never null: see computeModelMeanStdDev()
    float b = ( (float) pixel[ 0 ] - bdist * mean[ 0 ] ) / stddev[ 0 ];
    float g = ( (float) pixel[ 1 ] - bdist * mean[ 1 ] ) / stddev[ 1 ];
    float r = ( (float) pixel[ 2 ] - bdist * mean[ 2 ] ) / stddev[ 2 ];
    return sqrt( r * r + g * g + b * b );
}


void had::LCM::selectThresholdMatrix( const cv::Mat& mat,
                                      const float detection_rate,  // example: .99 for 99%
                                            float *left,
                                            float *right )
{
    CV_Assert( mat.type() == CV_32F );

    // Convert the matrix into an array
    int size = mat.rows * mat.cols;
    float* array = new float[ size ];
    for( int y = 0; y < mat.rows; ++y )
    {
        for( int x = 0; x < mat.cols; ++x )
        {
            array[ x + y * mat.cols ] = mat.at<float>( y, x );
        }
    }

    // Get the values at rate and (1 - detection_rate)
    int index_left  = (1 - detection_rate) * (float) size;
    int index_right = detection_rate       * (float) size;
    std::sort( array, array + size + 1 );
    *left  = array[ index_left ];
    *right = array[ index_right ];


    if( _trace )
    {
        std::cerr << "size: " << size << " | "
                  << "index_left: " << index_left << " " << "index_right: " << index_right << " " 
                  << array[ index_left ] << " " << array[ index_right ]
                  << std::endl;
    }

    delete[] array;
}


void had::LCM::selectThresholds( const float    detection_rate, // example: .99 for 99%
                                 const cv::Mat& bdist_norm,
                                 const cv::Mat& cdist_norm,
                                       float*   threshold_cdist,
                                       float*   threshold_bdist_left,
                                       float*   threshold_bdist_right )
{
    // See Horprasert et al., 1999, Section 4.3
    float dummy;
    selectThresholdMatrix( cdist_norm, detection_rate, &dummy, threshold_cdist );
    selectThresholdMatrix( bdist_norm, detection_rate, threshold_bdist_left, threshold_bdist_right );
}


void had::LCM::classify( const cv::Mat& image,
                               cv::Mat& out_classification )
{
    // See Horprasert et al., 1999, Eq 11
    cv::Mat bdist_norm, cdist_norm;
    computeNormalizedDistortions( image, bdist_norm, cdist_norm );

    if( _trace )
    {
        std::cerr << "Thresholds: cdist=" << _threshold_cdist << " "
                  << "bdist_left=" << _threshold_bdist_left << " " 
                  << "bdist_right=" << _threshold_bdist_right << " " 
                  << std::endl;
    }

    // CD_i > T_cd
    cv::Mat mask_foreground;
    cv::threshold( cdist_norm,
                   mask_foreground,
                   _threshold_cdist,
                   had::LCM::FOREGROUND,
                   cv::THRESH_BINARY );

    // alpha_i > T_alpha2
    cv::Mat mask_background_left;
    cv::threshold( bdist_norm,
                   mask_background_left,
                   _threshold_bdist_left,
                   had::LCM::BACKGROUND,
                   cv::THRESH_BINARY );

    // alpha_i < T_alpha1
    cv::Mat mask_background_right;
    cv::threshold( - bdist_norm,
                   mask_background_right,
                   - _threshold_bdist_right,
                   had::LCM::BACKGROUND,
                   cv::THRESH_BINARY );

    // Combine the two background masks
    cv::Mat mask_background = mask_background_left & mask_background_right;

    // alpha_i < 0
    cv::Mat mask_shadow;
    cv::threshold( - bdist_norm,
                   mask_shadow,
                   0,
                   had::LCM::SHADOW,
                   cv::THRESH_BINARY );

    cv::Mat mask_foreground_8u, mask_background_8u, mask_shadow_8u;
    mask_foreground.convertTo( mask_foreground_8u, CV_8UC1 );
    mask_background.convertTo( mask_background_8u, CV_8UC1 );
    mask_shadow.convertTo( mask_shadow_8u, CV_8UC1 );

    if( _trace )
    {
        showImage( "foreground", (mask_foreground_8u - 3) * 255, 6, 0 );
        showImage( "shadow", (mask_shadow_8u - 1) * 255, 6, 1 );
        showImage( "background", mask_background_8u * 255, 6, 2 );
    }

    // Now use the different masks to create the classification
    out_classification = cv::Mat( image.size(),
                                  CV_8UC1,
                                  cv::Scalar( had::LCM::HIGHLIGHT ) );

    mask_shadow_8u.copyTo( out_classification, mask_shadow_8u );
    mask_background_8u.copyTo( out_classification, mask_background_8u );
    mask_foreground_8u.copyTo( out_classification, mask_foreground_8u );
}


void had::LCM::classificationToImage( const cv::Mat& classification,
                                          cv::Mat& out_image )
{
    out_image = cv::Mat( classification.size(), CV_8UC3, cv::Scalar::all( 0 ) );
    for( int y = 0; y < classification.rows; ++y )
    {
        for( int x = 0; x < classification.cols; ++x )
        {
            unsigned char type = classification.at<unsigned char>( y, x );
            if( type == had::LCM::FOREGROUND )
            {
                out_image.at<cv::Vec3b>( y, x )[ 0 ] = 255; // blue
            }
            else if( type == had::LCM::BACKGROUND )
            {
                out_image.at<cv::Vec3b>( y, x )[ 1 ] = 255; // green
            }
            else if( type == had::LCM::SHADOW )
            {
                out_image.at<cv::Vec3b>( y, x )[ 2 ] = 255; // red
            }
            // else: type == had::LCM::HIGHLIGHT => black
        }
    }
}

