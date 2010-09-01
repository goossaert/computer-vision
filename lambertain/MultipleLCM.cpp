#include "MultipleLCM.hpp"

void had::MultipleLCM::computeModelMeanStdDev( const vector<cv::Mat>& images )
                        
{
    // See Horprasert et al., 1999, Sections 4.1 and 7, and Eq. 4
    
    // Prepare matrices
    _mean       = cv::Mat( images[ 0 ].size(), CV_32FC3, cv::Scalar::all( 0 ) );
    _stddev     = cv::Mat( images[ 0 ].size(), CV_32FC3, cv::Scalar::all( 0 ) );
    _brightness = cv::Mat( images[ 0 ].size(), CV_32FC3, cv::Scalar::all( 0 ) );
    
    cv::Mat pixels = cv::Mat( images.size(), 1, CV_8UC3, cv::Scalar::all( 0 ) );
    cv::Scalar mean, stddev;

    for( int y = 0; y < images[ 0 ].rows; ++y )
    {
        for( int x = 0; x < images[ 0 ].cols; ++x )
        {
            // Prepare the pixels from the different images
            for( unsigned int id_image = 0; id_image < images.size(); ++id_image )
            {
                pixels.at<cv::Vec3b>( id_image, 0 ) = images[ id_image ].at<cv::Vec3b>( y, x );
            }
 
            cv::meanStdDev( pixels, mean, stddev );
            for( int id = 0; id < 3; ++id )
                if( stddev[ id ] == 0 ) stddev[ id ] = 1;

            float denom = computeBrightnessDenominator( mean, stddev );
            for( int id = 0; id < 3; ++id )
            {
                _brightness.at<cv::Vec3f>( y, x )[ id ] = mean[ id ] / ( denom * stddev[ id ] * stddev[ id ] );
                _mean.at<cv::Vec3f>( y, x )[ id ] = mean[ id ];
                _stddev.at<cv::Vec3f>( y, x )[ id ] = stddev[ id ];
            }
        }
    }
}


void had::MultipleLCM::computeModel( const vector<cv::Mat>& images )
{
    // See Horprasert et al., 1999, Section 4.1
    computeModelMeanStdDev( images );
    computeVariations( images );

    cv::Mat bdist_norm, cdist_norm;
    computeNormalizedDistortions( images, bdist_norm, cdist_norm );

    selectThresholds( _detection_rate,
                      bdist_norm,
                      cdist_norm,
                      &_threshold_cdist,
                      &_threshold_bdist_left,
                      &_threshold_bdist_right
                    );
}


float had::MultipleLCM::computeBrightnessDistortion( const cv::Mat& image,
                                                              int y,
                                                              int x )
{
    cv::Scalar brightness( _brightness.at<cv::Vec3f>( y, x )[ 0 ],
                           _brightness.at<cv::Vec3f>( y, x )[ 1 ],
                           _brightness.at<cv::Vec3f>( y, x )[ 2 ] );
    return LCM::computeBrightnessDistortion( image.at<cv::Vec3b>( y, x ),
                                             brightness );
}


float had::MultipleLCM::computeChromacityDistortion( const cv::Mat& image,
                                                              int y,
                                                              int x,
                                                              float bdist )
{
    cv::Scalar mean( _mean.at<cv::Vec3f>( y, x )[ 0 ],
                     _mean.at<cv::Vec3f>( y, x )[ 1 ],
                     _mean.at<cv::Vec3f>( y, x )[ 2 ] );
    cv::Scalar stddev( _stddev.at<cv::Vec3f>( y, x )[ 0 ],
                       _stddev.at<cv::Vec3f>( y, x )[ 1 ],
                       _stddev.at<cv::Vec3f>( y, x )[ 2 ] );
    return LCM::computeChromacityDistortion( image.at<cv::Vec3b>( y, x ),
                                             mean,
                                             stddev,
                                             bdist );
}


void had::MultipleLCM::computeVariations( const vector<cv::Mat>& images )
{
    // See Horprasert et al., 1999, Section 4.1

    _bdist_variation = cv::Mat( images[ 0 ].size(), CV_32F, cv::Scalar::all( 0 ) );
    _cdist_variation = cv::Mat( images[ 0 ].size(), CV_32F, cv::Scalar::all( 0 ) );
    int nb_pixels = images.size();

    for( int y = 0; y < images[ 0 ].rows; ++y )
    {
        for( int x = 0; x < images[ 0 ].cols; ++x )
        {
            // The variations are computed for each pixel position
            // and accross all training images
            float bdist_sum = 0;
            float cdist_sum = 0;

            for( unsigned int id_image = 0; id_image < images.size(); ++id_image )
            {
                float bdist_current = computeBrightnessDistortion( images[ id_image ],
                                                                   y,
                                                                   x );

                float cdist_current = computeChromacityDistortion( images[ id_image ],
                                                                   y,
                                                                   x,
                                                                   bdist_current );

                bdist_sum += ( bdist_current - 1 ) * ( bdist_current - 1 );
                cdist_sum += cdist_current * cdist_current;
            }

            // See Horprasert et al., 1999, Eqs. 7 and 8
            _bdist_variation.at<float>( y, x ) = sqrt( bdist_sum / (float) nb_pixels );
            _cdist_variation.at<float>( y, x ) = sqrt( cdist_sum / (float) nb_pixels );
        }
    }
}


void had::MultipleLCM::computeNormalizedDistortions( const cv::Mat& image,
                                                           cv::Mat& out_bdist_norm,
                                                           cv::Mat& out_cdist_norm )
{
    vector<cv::Mat> images;
    images.push_back( image );
    computeNormalizedDistortions( images, out_bdist_norm, out_cdist_norm );
}



void had::MultipleLCM::computeNormalizedDistortions( const vector<cv::Mat>& images,
                                                           cv::Mat& out_bdist_norm,
                                                           cv::Mat& out_cdist_norm )
{
    // See Horprasert et al., 1999, Eqs. 9 and 10
    int cols = images[ 0 ].cols;
    int rows = images[ 0 ].rows;
    out_bdist_norm = cv::Mat( rows, cols * images.size(), CV_32F, cv::Scalar::all( 0 ) );
    out_cdist_norm = cv::Mat( rows, cols * images.size(), CV_32F, cv::Scalar::all( 0 ) );
    for( unsigned int id_image = 0; id_image < images.size(); ++id_image )
    {
        cv::Mat bdist_image = out_bdist_norm.colRange( id_image * cols, ( id_image + 1 ) * cols );
        cv::Mat cdist_image = out_cdist_norm.colRange( id_image * cols, ( id_image + 1 ) * cols );
        for( int y = 0; y < rows; ++y )
        {
            for( int x = 0; x < cols; ++x )
            {
                float bdist_current = computeBrightnessDistortion( images[ id_image ], y, x );
                float cdist_current = computeChromacityDistortion( images[ id_image ], y, x, bdist_current );
                bdist_image.at<float>( y, x ) = (bdist_current - 1) / _bdist_variation.at<float>( y, x );
                cdist_image.at<float>( y, x ) = cdist_current / _cdist_variation.at<float>( y, x );
            }
        }
    }
}
