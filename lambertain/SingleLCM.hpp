#ifndef HAD_SINGLE_LCM_HPP
#define HAD_SINGLE_LCM_HPP

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
* @brief Single image Lambertain Color Model.
*
* 
*
*/
/* ----------------------------------------------------------------------------*/
class SingleLCM: public LCM
{
private:

    cv::Scalar _mean;             //!< Mean of the background pixels in the model.
    cv::Scalar _stddev;           //!< Standard deviation of the background pixels in the model.
    cv::Scalar _brightness;       //!< Brighness denominator used to speed up computations.
    float      _bdist_variation;  //!< Brightness distortion variation
    float      _cdist_variation;  //!< Chromacity distortion variation

    /* ----------------------------------------------------------------------------*/
    /** 
    * @brief Compute the mean and standard deviation of an image, only using
    * the pixels in the mask.
    *
    * See Horprasert et al., 1999, Sections 4.1 and 7
    * See Horprasert et al., 1999, Eq. 4
    * 
    * @param image Input image (8-bit 3-channel image, CV_8UC3).
    * @param mask Input mask (8-bit 1-channel image, CV_8UC1).
    */
    /* ----------------------------------------------------------------------------*/
    void computeModelMeanStdDev( const cv::Mat& image,
                                 const cv::Mat& mask );

    /* ----------------------------------------------------------------------------*/
    /** 
    * @brief Compute the Lambertain Color Model based on an image and using only the
    * pixels in the mask as training background pixels.
    *
    * See Horprasert et al., 1999, Section 4.1
    * 
    * @param image Input image (8-bit 3-channel image, CV_8UC3).
    * @param mask Input mask (8-bit 1-channel image, CV_8UC1).
    */
    /* ----------------------------------------------------------------------------*/
    void computeModel( const cv::Mat& image,
                       const cv::Mat& mask );

    /* ----------------------------------------------------------------------------*/
    /** 
    * @brief Compute the variations of the brightness and chromaticity distributions.
    *
    * See Horprasert et al., 1999, Section 4.1
    * See Yacoob and Davis, 2006, Section 2.2
    * 
    * @param image Input image (8-bit 3-channel image, CV_8UC3).
    * @param mask Input mask (8-bit 1-channel image, CV_8UC1).
    */
    /* ----------------------------------------------------------------------------*/
    void computeVariations( const cv::Mat& image,
                            const cv::Mat& mask );

    virtual void computeNormalizedDistortions( const cv::Mat& image,
                                                     cv::Mat& out_bdist_norm,
                                                     cv::Mat& out_cdist_norm );

    virtual void computeNormalizedDistortions( const vector<cv::Mat>& image,
                                                     cv::Mat& out_bdist_norm,
                                                     cv::Mat& out_cdist_norm );

protected:
    virtual float computeBrightnessDistortion( const cv::Mat& image,
                                               int y,
                                               int x );

    virtual float computeChromacityDistortion( const cv::Mat& image,
                                               int y,
                                               int x,
                                               float bdist );
public:
    /* ----------------------------------------------------------------------------*/
    /** 
    * @brief Constructor.
    * 
    * @param image Training image (8-bit 3-channel image, CV_8UC3).
    * @param detection_rate Detection rate (ex: 95% is .95).
    * @param regions Vector of rectangles used to indicate which areas are used
    * as training background pixels.
    */
    /* ----------------------------------------------------------------------------*/
    SingleLCM( const cv::Mat&          image, 
               const float             detection_rate,
               const vector<cv::Rect>& regions,
               const bool              trace = false )
    : LCM( detection_rate, trace )
    {
        cv::Mat mask( image.size(), CV_8UC1, cv::Scalar( 0 ) );
        fillRectangles( mask, regions, cv::Scalar( 1 ) );
        computeModel( image, mask );
    }

    /* ----------------------------------------------------------------------------*/
    /** 
    * @brief 
    * 
    * @param image Training image (8-bit 3-channel image, CV_8UC3).
    * @param detection_rate Detection rate (ex: 95% is .95).
    * @param mask Mask used to indicate which pixels are used as training
    * background pixels (8-bit 1-channel image, CV_8UC1).
    */
    /* ----------------------------------------------------------------------------*/
    SingleLCM( const cv::Mat& image, 
               const float    detection_rate,
               const cv::Mat& mask,
               const bool     trace = false )
    : LCM( detection_rate, trace )
    {
        computeModel( image, mask );
    }

    /* ----------------------------------------------------------------------------*/
    /** 
    * @brief Destructor.
    */
    /* ----------------------------------------------------------------------------*/
    virtual ~SingleLCM() {}
};

}

#endif // HAD_SINGLE_LCM_HPP
