// (c)2010 - Emmanuel Goossaert
// Under GNU License 3.0
#include "SingleLCM.hpp"

void had::SingleLCM::computeModelMeanStdDev( const cv::Mat& image,
                                             const cv::Mat& mask )
{
    // See Horprasert et al., 1999, Sections 4.1 and 7
    // See Horprasert et al., 1999, Eq. 4
    
    // Pre-compute the brightness denominator for future calculations
    cv::meanStdDev( image, _mean, _stddev, mask );
    for( int id = 0; id < 3; ++id )
        if( _stddev[ id ] == 0 ) _stddev[ id ] = 1;

    // There is possibly a typo in Yacoob and Davis, that I have fixed.
    // In order to get the formula as they presented it, just square
    // the mean value below.
    float denom = computeBrightnessDenominator( _mean, _stddev );
    for( int id = 0; id < 3; ++id )
        _brightness[ id ] = _mean[ id ] / ( denom * _stddev[ id ] * _stddev[ id ] );
}


void had::SingleLCM::computeModel( const cv::Mat& image,
                                   const cv::Mat& mask )
{
    // See Horprasert et al., 1999, Section 4.1
    computeModelMeanStdDev( image, mask );
    computeVariations( image, mask );

    if( _trace )
    {
        std::cerr << "mean: B=" << _mean[ 0 ] << " G=" << _mean[ 1 ] << " R=" << _mean[ 2 ] << std::endl;
        std::cerr << "stddev: B=" << _stddev[ 0 ] << " G=" << _stddev[ 1 ] << " R=" << _stddev[ 2 ] << std::endl;
        showImage( "lcm mask", mask * 255, 5, 3 );
    }

    cv::Mat bdist_norm, cdist_norm;
    computeNormalizedDistortions( image, bdist_norm, cdist_norm );

    selectThresholds( _detection_rate,
                      bdist_norm,
                      cdist_norm,
                      &_threshold_cdist,
                      &_threshold_bdist_left,
                      &_threshold_bdist_right
                    );
}


float had::SingleLCM::computeBrightnessDistortion( const cv::Mat& image,
                                                              int y,
                                                              int x )
{
    return LCM::computeBrightnessDistortion( image.at<cv::Vec3b>( y, x ),
                                             _brightness );
}


float had::SingleLCM::computeChromacityDistortion( const cv::Mat& image,
                                                              int y,
                                                              int x,
                                                              float bdist )
{
    return LCM::computeChromacityDistortion( image.at<cv::Vec3b>( y, x ),
                                             _mean,
                                             _stddev,
                                             bdist );
}


void had::SingleLCM::computeVariations( const cv::Mat& image,
                                                   const cv::Mat& mask )
{
    // See Horprasert et al., 1999, Section 4.1
    // See Yacoob and Davis, 2006, Section 2.2
    CV_Assert( image.type() == CV_8UC3 && mask.type() == CV_8UC1 && image.size() == mask.size() );

    float bdist_variation = 0;
    float cdist_variation = 0;

    int nb_pixels = 0;
    for( int y = 0; y < image.rows; ++y )
    {
        for( int x = 0; x < image.cols; ++x )
        {
            // Consider only the pixels in the training regions, which is the modification
            // of Yacoob and Davis, 2006, to the model developed by Horprasert et al., 1999.
            if( ! mask.at<unsigned char>( y, x ) )
                continue;

            float bdist_current = computeBrightnessDistortion( image, y, x );
            float cdist_current = computeChromacityDistortion( image, y, x, bdist_current );
            bdist_variation += ( bdist_current - 1 ) * ( bdist_current - 1 );
            cdist_variation += cdist_current * cdist_current;
            ++nb_pixels;
        }
    }

    if( nb_pixels > 0 )
    {
        // See Horprasert et al., 1999, Eqs. 7 and 8
        _bdist_variation = sqrt( bdist_variation / (float) nb_pixels );
        _cdist_variation = sqrt( cdist_variation / (float) nb_pixels );
    }
    else
    {
        // Set to 1 to avoid error when dividing by these variations
        _bdist_variation = 1;
        _cdist_variation = 1;
    }

    if( _trace )
    {
        std::cerr << "nb pixels: " << nb_pixels << " "
                  << "bdist_var: " << _bdist_variation << " "
                  << "cdist_var: " << _cdist_variation << " "
                  << std::endl;
    }
}


void had::SingleLCM::computeNormalizedDistortions( const vector<cv::Mat>& images,
                                                   cv::Mat& out_bdist_norm,
                                                   cv::Mat& out_cdist_norm )
{
    computeNormalizedDistortions( images[ 0 ], out_bdist_norm, out_cdist_norm );
}


void had::SingleLCM::computeNormalizedDistortions( const cv::Mat& image,
                                                   cv::Mat& out_bdist_norm,
                                                   cv::Mat& out_cdist_norm )
{
    // See Horprasert et al., 1999, Eq. 9 and 10
   
    out_bdist_norm = cv::Mat( image.size(), CV_32F );
    out_cdist_norm = cv::Mat( image.size(), CV_32F );
    for( int y = 0; y < image.rows; ++y )
    {
        for( int x = 0; x < image.cols; ++x )
        {
            float bdist_current = computeBrightnessDistortion( image, y, x );
            float cdist_current = computeChromacityDistortion( image, y, x,  bdist_current );
            out_bdist_norm.at<float>( y, x ) = (bdist_current - 1) / _bdist_variation;
            out_cdist_norm.at<float>( y, x ) = cdist_current / _cdist_variation;
        }
    }
}

