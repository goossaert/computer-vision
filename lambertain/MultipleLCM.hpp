#ifndef HAD_MULTIPLE_LCM_HPP
#define HAD_MULTIPLE_LCM_HPP

#include <iostream>
#include <algorithm>
#include <vector>
using std::vector;
#include <string>
using std::string;

#include <cv.h>
#include <highgui.h>

#include "LCM.hpp"

namespace had {

/* ----------------------------------------------------------------------------*/
/** 
* @brief Multiple image Lambertain Color Model.
*
*
*/
/* ----------------------------------------------------------------------------*/
class MultipleLCM: public LCM
{
private:

    cv::Mat _mean;              //!< Mean of the background pixels in the model.
    cv::Mat _stddev;            //!< Standard deviation of the background pixels in the model.
    cv::Mat _brightness;        //!< Brighness denominator used to speed up computations.
    cv::Mat _bdist_variation;   //!< Brightness distortion variation
    cv::Mat _cdist_variation;   //!< Chromacity distortion variation

    /* ----------------------------------------------------------------------------*/
    /** 
    * @brief Compute the mean and standard deviation of a set of images.
    * 
    * See Horprasert et al., 1999, Sections 4.1 and 7, and Eq. 4
    *
    * @param images Vector of input images (8-bit 3-channel images, CV_8UC3).
    */
    /* ----------------------------------------------------------------------------*/
    void computeModelMeanStdDev( const vector<cv::Mat>& images );


    /* ----------------------------------------------------------------------------*/
    /** 
    * @brief Compute the Lambertain Color Model based on a set of images. All the
    * pixels in each image are used as training background pixels.
    *
    * See Horprasert et al., 1999, Section 4.1
    * 
    * @param images Vector of input images (8-bit 3-channel images, CV_8UC3).
    */
    /* ----------------------------------------------------------------------------*/
    void computeModel( const vector<cv::Mat>& images );

    /* ----------------------------------------------------------------------------*/
    /** 
    * @brief Compute the variations of the brightness and chromaticity distributions.
    *
    * See Horprasert et al., 1999, Section 4.1
    * 
    * @param images Vector of input images (8-bit 3-channel images, CV_8UC3).
    */
    /* ----------------------------------------------------------------------------*/
    void computeVariations( const vector<cv::Mat>& images );

    virtual float computeBrightnessDistortion( const cv::Mat& image,
                                       int y,
                                       int x );

    virtual float computeChromacityDistortion( const cv::Mat& image,
                                       int y,
                                       int x,
                                       float bdist );

    virtual void computeNormalizedDistortions( const vector<cv::Mat>& images,
                                                     cv::Mat& out_bdist_norm,
                                                     cv::Mat& out_cdist_norm );

    virtual void computeNormalizedDistortions( const cv::Mat& image,
                                                     cv::Mat& out_bdist_norm,
                                                     cv::Mat& out_cdist_norm );

public:
    /* ----------------------------------------------------------------------------*/
    /** 
    * @brief Constructor.
    * 
    * @param images Vector of training images (8-bit 3-channel images, CV_8UC3).
    * @param detection_rate Detection rate (ex: 95% is .95).
    */
    /* ----------------------------------------------------------------------------*/
    MultipleLCM( const vector<cv::Mat>& images,
                 const float            detection_rate,
                 const bool             trace = false )
    : LCM( detection_rate, trace )
    {
        if( images.empty() )
        {
            std::cerr << "ERROR: no input images!" << std::endl;
            exit( 0 );
        }

        computeModel( images );
    }

    /* ----------------------------------------------------------------------------*/
    /** 
    * @brief Destructor.
    */
    /* ----------------------------------------------------------------------------*/
    virtual ~MultipleLCM() {}
};


}

#endif // HAD_MULTIPLE_LCM_HPP

