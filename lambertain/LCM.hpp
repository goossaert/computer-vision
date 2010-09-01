// (c)2010 - Emmanuel Goossaert
// Under GNU License 3.0
#ifndef HAD_LCM_HPP
#define HAD_LCM_HPP

#include <iostream>
#include <algorithm>
#include <vector>
using std::vector;
#include <string>
using std::string;

#include <cv.h>
#include <highgui.h>

namespace had {

/* ----------------------------------------------------------------------------*/
/** 
* @brief Lambertain Color Model.
*
* This is the implementation of the Lambertain Color Model, based on the paper
* "A Statistical Approach for Real-time Robust Background Subtraction
* and Shadow Detection" by Horprasert et al., 1999, and on the paper
* "Detection and Analysis of Hair" by Yacoob and Davis, 2006.
*
* The main difference between the model presented by Horprasert et al., 1999,
* and Yacoob and Davis, 2006, is that in the former the model is computed over
* the pixels in a set of N training images, whereas in the latter the model is
* computed over the pixels of a defined training region in a single input image.
*
* The Clustering Detection Elimination (Horprasert et al., 1999, Section 5),
* has not been implemented yet.
*
* The constants used to classify the pixels are as follow (this description is
* a citation from Horprasert et al., 1999, Section 4.2):
*
*  - BACKGROUND: Original background, if it has both brightness and
*    chromaticity similar to those of the same pixel in the background image.
*  - SHADOW: Shaded background or shadow, if it has similar chromaticity but
*    lower brightness than those of the same pixel in the background image.
*    This is based on the notion of the shadow as a semi-transparent region
*    in the image, which retains a representation of the underlying surface
*    pattern, texture or color value.
*  - HIGHLIGHT: Highlighted background, if it has similar chromaticity but
*    higher brightness than the background image.
*  - FOREGROUND: Moving foreground object, if the pixel has chromaticity
*    different from the expected values in the background image.
*/
/* ----------------------------------------------------------------------------*/
class LCM
{
protected:
    float      _detection_rate;         //!< Percentage of background pixels to (ex: .95 means 95%)
    float      _threshold_cdist;        //!< Contrast distortion threshold (computed automatically)
    float      _threshold_bdist_left;   //!< Left brightness distortion threshold (computed automatically)
    float      _threshold_bdist_right;  //!< Right brightness distortion threshold (computed automatically)
    bool       _trace;                  //!< If true, show debugging values and images

    /* ----------------------------------------------------------------------------*/
    /** 
    * @brief Fill a rectangular area with a Scalar in image.
    * 
    * @param io_image 8-bit image, either 3-channel or 1-channel, that will be
    * modified in-place.
    * @param rect Rectangular area to fill.
    * @param value Value to fill the rectangle area with, either 3-channel or
    * 1-channel, depending on the image.
    */
    /* ----------------------------------------------------------------------------*/
    void fillRectangle( cv::Mat& io_image,
                        const cv::Rect& rect,
                        const cv::Scalar value );

    /* ----------------------------------------------------------------------------*/
    /** 
    * @brief Fill a rectangular area with a Scalar in image.
    * 
    * @param io_image 8-bit image, either 3-channel or 1-channel, that will be
    * modified in-place.
    * @param rectangles Vector of Rectangular areas to fill.
    * @param value Value to fill the rectangle areas with, either 3-channel or
    * 1-channel, depending on the image.
    */
    /* ----------------------------------------------------------------------------*/
    void fillRectangles( cv::Mat& io_image,
                         const vector<cv::Rect>& rectangles,
                         const cv::Scalar value );

    /* ----------------------------------------------------------------------------*/
    /** 
    * @brief Show an image in the cell of a grid system, for debugging purposes.
    *
    * The screen is divided into cells of equal size, and the given image is shown
    * into a cell of that grid. The col and row parameters determine which cell
    * it is. The size of the cell (and therefore of the image) is determined inside
    * of the method, to maintain alignment.
    * 
    * @param name Name of the window.
    * @param image Image to show (any format will work).
    * @param col Column of the cell in which the image will be shown (starting at 0).
    * @param row Row of the cell in which the image will be shown (starting at 0).
    */
    /* ----------------------------------------------------------------------------*/
    void showImage( const string name, const cv::Mat& image, int col, int row );

    /* ----------------------------------------------------------------------------*/
    /** 
    * @brief Select thresholds based upon values in a matrix.
    * 
    * @param mat The matrix (CV_32F) on which the thresholds need to be computed.
    * @param detection_rate Detection rate (ex: 95% is .95)
    * @param out_left Computed left threshold.
    * @param out_right Computed right threshold.
    */
    /* ----------------------------------------------------------------------------*/
    void selectThresholdMatrix( const cv::Mat& mat,
                                const float detection_rate,
                                      float *out_left,
                                      float *out_right );

    /* ----------------------------------------------------------------------------*/
    /** 
    * @brief Select thresholds for the color model.
    * 
    * @param detection_rate Detection rate (ex: 95% is .95)
    * @param bdist_norm Normalized brightness distortion distribution.
    * @param cdist_norm Normalize chromaticity distortion distribution.
    * @param out_threshold_cdist Computed chromaticity distortion threshold.
    * @param out_threshold_bdist_left Computed left brightness distortion threshold.
    * @param out_threshold_bdist_right Computed right brightness distortion threshold.
    */
    /* ----------------------------------------------------------------------------*/
    void selectThresholds( const float    detection_rate,
                           const cv::Mat& bdist_norm,
                           const cv::Mat& cdist_norm,
                                 float*   out_threshold_cdist,
                                 float*   out_threshold_bdist_left,
                                 float*   out_threshold_bdist_right );

    /* ----------------------------------------------------------------------------*/
    /** 
    * @brief Compute the brightness denominator to help calculations.
    *
    * See Horprasert et al., 1999, Eq. 5
    * See Yacoob and Davis, 2006, Eq. 3
    * 
    * @param mean Mean of the training pixels.
    * @param stddev Standard deviation of the training pixels.
    * 
    * @return The brightness denominator.
    */
    /* ----------------------------------------------------------------------------*/
    float computeBrightnessDenominator( const cv::Scalar& mean,
                                        const cv::Scalar& stddev );

    /* ----------------------------------------------------------------------------*/
    /** 
    * @brief Compute the brightness distortion of a pixel.
    *
    * See Horprasert et al., 1999, Eq. 5
    * See Yacoob and Davis, 2006, Eq. 3
    * 
    * @param pixel Pixel (8-bit 3-channel Scalar).
    * @param brightness Brightness of the pixel.
    * 
    * @return The brightness distortion of the given pixel.
    */
    /* ----------------------------------------------------------------------------*/
    float computeBrightnessDistortion( const cv::Vec3b& pixel,
                                       const cv::Scalar& brightness );

    /* ----------------------------------------------------------------------------*/
    /** 
    * @brief Compute the brightness distortion of the pixel in a given image.
    *
    * This method is re-implemented by the actual models.
    * See Horprasert et al., 1999, Eq. 5
    * See Yacoob and Davis, 2006, Eq. 3
    * 
    * @param image Input image (8-bit 3-channel image, CV_8UC3)
    * @param y Y-coordinate of the pixel whose brightness distortion needs
    * to be computed.
    * @param x X-coordinate of the pixel whose brightness distortion needs
    * to be computed.
    * 
    * @return The brightness distortion of the given pixel.
    */
    /* ----------------------------------------------------------------------------*/
    virtual float computeBrightnessDistortion( const cv::Mat& image,
                                               int y,
                                               int x ) = 0;

    /* ----------------------------------------------------------------------------*/
    /** 
    * @brief Compute the chromaticity distortion of a pixel.
    *
    * See Horprasert et al., 1999, Eq. 6
    * See Yacoob and Davis, 2006, Eq. 4
    * 
    * @param pixel Pixel (8-bit 3-channel Scalar).
    * @param mean Mean of the training pixels (8-bit 3-channel Scalar).
    * @param stddev Standard deviation of the training pixels (8-bit 3-channel Scalar).
    * @param bdist Brightness distortion of the training pixels.
    * 
    * @return The chromaticity distortion of the given pixel.
    */
    /* ----------------------------------------------------------------------------*/
    float computeChromacityDistortion( const cv::Vec3b&  pixel,
                                       const cv::Scalar& mean,
                                       const cv::Scalar& stddev,
                                             float       bdist );

    /* ----------------------------------------------------------------------------*/
    /** 
    * @brief Compute the chromaticity distortion of the pixel in a given image.
    *
    * This method is re-implemented by the actual models.
    * See Horprasert et al., 1999, Eq. 6
    * See Yacoob and Davis, 2006, Eq. 4
    *
    * @param image Input image (8-bit 3-channel image, CV_8UC3)
    * @param y Y-coordinate of the pixel whose brightness distortion needs
    * to be computed.
    * @param x X-coordinate of the pixel whose brightness distortion needs
    * to be computed.
    * @param bdist Brightness distortion of the training pixels.
    * 
    * @return The chromaticity distortion of the given pixel.
    */
    /* ----------------------------------------------------------------------------*/
    virtual float computeChromacityDistortion( const cv::Mat& image,
                                                     int      y,
                                                     int      x,
                                                     float    bdist ) = 0;

    /* ----------------------------------------------------------------------------*/
    /** 
    * @brief Compute normalized brightness and chromaticity distortion distributions
    * based upon a single input image.
    * 
    * @param image Input image used to compute the distributions (8-bit 3-channel
    * image, CV_8UC3).
    * @param out_bdist_norm Computed brightness distortion distribution.
    * @param out_cdist_norm Computed chromaticity distortion distribution.
    */
    /* ----------------------------------------------------------------------------*/
    virtual void computeNormalizedDistortions( const cv::Mat& image,
                                                     cv::Mat& out_bdist_norm,
                                                     cv::Mat& out_cdist_norm ) = 0;

    /* ----------------------------------------------------------------------------*/
    /** 
    * @brief Compute normalized brightness and chromaticity distortion distributions
    * based upon a single input image.
    * 
    * @param images Vector of input images used to compute the distributions (8-bit
    * 3-channel images, CV_8UC3).
    * @param out_bdist_norm Computed brightness distortion distribution.
    * @param out_cdist_norm Computed chromaticity distortion distribution.
    */
    /* ----------------------------------------------------------------------------*/
    virtual void computeNormalizedDistortions( const vector<cv::Mat>& image,
                                                     cv::Mat& out_bdist_norm,
                                                     cv::Mat& out_cdist_norm ) = 0;

public:
    /* ----------------------------------------------------------------------------*/
    /** 
    * @brief Constructor.
    * 
    * @param detection_rate Detection rate (ex: 95% is .95).
    */
    /* ----------------------------------------------------------------------------*/
    LCM( const float detection_rate, const bool trace = false )
    : _detection_rate( detection_rate ), _trace( trace )
    {
    }

    /* ----------------------------------------------------------------------------*/
    /** 
    * @brief Destructor.
    */
    /* ----------------------------------------------------------------------------*/
    virtual ~LCM() {}

    /* ----------------------------------------------------------------------------*/
    /** 
    * @brief Classify the pixels of an input image based on a model.
    * 
    * @param image Input image (8-bit 3-channel image, CV_8UC3).
    * @param out_classification Computed classification image (8-bit 1-channel image,
    * CV_8UC1).
    */
    /* ----------------------------------------------------------------------------*/
    void classify( const cv::Mat& image,
                         cv::Mat& out_classification );

    /* ----------------------------------------------------------------------------*/
    /** 
    * @brief Show a classification image that has been outputed by classify().
    *
    * The different classes of pixels are shown with different colors. The legend
    * for this colors is as follow:
    *
    * - BLUE:  foreground
    * - GREEN: background
    * - RED:   background shadow
    * - BLACK: background highlight
    *
    * @param classification Classification image (8-bit 1-channel image, CV_8UC1).
    * @param out_image Output image that can be shown. 
    */
    /* ----------------------------------------------------------------------------*/
    void classificationToImage( const cv::Mat& classification,
                                      cv::Mat& out_image );

    const static unsigned char BACKGROUND = 1; //!< Background pixel.
    const static unsigned char SHADOW     = 2; //!< Background shadow pixel.
    const static unsigned char HIGHLIGHT  = 3; //!< Background highlight pixel.
    const static unsigned char FOREGROUND = 4; //!< Foreground pixel.
};

}

#endif // HAD_LCM_HPP
