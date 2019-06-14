/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2012 The MITRE Corporation                                      *
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License");           *
 * you may not use this file except in compliance with the License.          *
 * You may obtain a copy of the License at                                   *
 *                                                                           *
 *     http://www.apache.org/licenses/LICENSE-2.0                            *
 *                                                                           *
 * Unless required by applicable law or agreed to in writing, software       *
 * distributed under the License is distributed on an "AS IS" BASIS,         *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 * See the License for the specific language governing permissions and       *
 * limitations under the License.                                            *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <opencv2/features2d/features2d.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV Key Point Detector
 * \br_link http://docs.opencv.org/modules/features2d/doc/common_interfaces_of_feature_detectors.html
 * \author Josh Klontz \cite jklontz
 */
class KeyPointDetectorTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(QString detector READ get_detector WRITE set_detector RESET reset_detector STORED false)
    BR_PROPERTY(QString, detector, "SIFT")

    Ptr<FeatureDetector> featureDetector;

    void init()
    {
        // todo: directly cast string to template/class
        std::string detectorName = detector.toStdString();
        if (detectorName == "BRISK")
        {
            featureDetector = BRISK::create();
        }
        else if (detectorName == "ORB")
        {
            featureDetector = ORB::create();
        }
        else if (detectorName == "MSER")
        {
            featureDetector = MSER::create();
        }
        else if (detectorName == "KAZE")
        {
            featureDetector = KAZE::create();
        }
        else if (detectorName == "AKAZE")
        {
            featureDetector = AKAZE::create();
        }
        else if (detectorName == "GFTTDetector")
        {
            featureDetector = GFTTDetector::create();
        }
        else if (detectorName == "FastFeatureDetector")
        {
            featureDetector = FastFeatureDetector::create();
        }
        else if (detectorName == "AgastFeatureDetector")
        {
            featureDetector = AgastFeatureDetector::create();
        }
        else if (detectorName == "SimpleBlobDetector")
        {
            featureDetector = SimpleBlobDetector::create();
        }

        if (featureDetector.empty())
            qFatal("Failed to create KeyPointDetector: %s", qPrintable(detector));
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        std::vector<KeyPoint> keyPoints;
        try {
            featureDetector->detect(src, keyPoints);
        } catch (...) {
            qWarning("Key point detection failed for file %s", qPrintable(src.file.name));
            dst.file.fte = true;
        }

        QList<Rect> rects;
        foreach (const KeyPoint &keyPoint, keyPoints)
            rects.append(Rect(keyPoint.pt.x-keyPoint.size/2, keyPoint.pt.y-keyPoint.size/2, keyPoint.size, keyPoint.size));
        dst.file.setRects(OpenCVUtils::fromRects(rects));
    }
};

BR_REGISTER(Transform, KeyPointDetectorTransform)

} // namespace br

#include "metadata/keypointdetector.moc"
