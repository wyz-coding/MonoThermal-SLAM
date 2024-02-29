#include "include/Lineextractor.h"
#include "include/ELSED.h"

namespace MonoThermal_SLAM
{
    LSDExtractor::LSDExtractor(const int nLineFeatures, const int nminLinelength)
    :nLineFeatures_(nLineFeatures), nminLineLength_(nminLinelength)
    {
        lsd_ = cv::line_descriptor::LSDDetector::createLSDDetector();
        lbd_ = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor( );

    }


    void LSDExtractor::operator()
            (const cv::Mat& im, std::vector<cv::line_descriptor::KeyLine> &line, cv::Mat& lbd_descr)
    {
        assert(0);
        float scale=1.2;
        int numOctaves=8;
        cv::Mat _mask;
        std::cout<<"here"<<std::endl;
        lsd_->detect(im, line, scale, numOctaves,_mask);
        lbd_->compute( im, line, lbd_descr );
    }




    FLDExtractor::FLDExtractor(int nLineFeatures, int nminLinelength)
    :nLineFeatures_(nLineFeatures), nminLineLength_(nminLinelength)
    {
        fld_ = cv::ximgproc::createFastLineDetector(nminLineLength_);
        lsd_ = cv::line_descriptor::LSDDetector::createLSDDetector();
        // fld_ = cv::createLineSegmentDetector();
        lbd_ = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor( );
    }

    void FLDExtractor::operator()(const cv::Mat &im, std::vector<cv::line_descriptor::KeyLine> &line, cv::Mat &lbd_descr)
    {
        cv::Mat img_clahe;
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(im, img_clahe);
        // img_clane=im;
        std::vector<cv::Vec4f> fld_lines_tmp;
        upm::ELSED elsed;
        
        // std::cout << "******************************************************" << std::endl;
        // std::cout << "******************* ELSED main demo ******************" << std::endl;
        // std::cout << "******************************************************" << std::endl;

        upm::Segments segs = elsed.detect(img_clahe);

        // fld_->detect( img_clane, fld_lines_tmp );
        fld_lines_tmp = segs;


        std::vector<cv::Vec4f> fld_lines;
        {
            //TODO add to yaml
            for( auto& fld_line:fld_lines_tmp )
            {
               bool outLeft  = fld_line[0] < 35.f  && fld_line[2] < 35.f;
               bool outRight = fld_line[0] > 600.f && fld_line[2] > 600.f;
               bool outUp    = fld_line[1] < 40.f  && fld_line[3] < 40.f;
               bool outDown  = fld_line[1] > 440.f && fld_line[3] > 440.f;
               if(outLeft || outRight || outUp || outDown){
                   continue;
               }
                fld_lines.push_back(fld_line);
            }
        }

        if(fld_lines.size() > nLineFeatures_)
        {
            std::sort( fld_lines.begin(), fld_lines.end(), sort_flines_by_length() );
            fld_lines.resize(nLineFeatures_);
        }


        // loop over lines object transforming into a vector<KeyLine>
        line.reserve(fld_lines.size());
        for( int i = 0; i < fld_lines.size(); i++ )
        {
            cv::line_descriptor::KeyLine kl;
            double octaveScale = 1.f;
            int    octaveIdx   = 0;

            kl.startPointX     = fld_lines[i][0] * octaveScale;
            kl.startPointY     = fld_lines[i][1] * octaveScale;
            kl.endPointX       = fld_lines[i][2] * octaveScale;
            kl.endPointY       = fld_lines[i][3] * octaveScale;

            kl.sPointInOctaveX = fld_lines[i][0];
            kl.sPointInOctaveY = fld_lines[i][1];
            kl.ePointInOctaveX = fld_lines[i][2];
            kl.ePointInOctaveY = fld_lines[i][3];

            kl.lineLength = (float) sqrt( pow( fld_lines[i][0] - fld_lines[i][2], 2 ) + pow( fld_lines[i][1] - fld_lines[i][3], 2 ) );

            kl.angle    = std::atan2( ( kl.endPointY - kl.startPointY ), ( kl.endPointX - kl.startPointX ) );
            kl.class_id = i;
            kl.octave   = octaveIdx;
            kl.size     = ( kl.endPointX - kl.startPointX ) * ( kl.endPointY - kl.startPointY );
            kl.pt       = cv::Point2f( ( kl.endPointX + kl.startPointX ) / 2, ( kl.endPointY + kl.startPointY ) / 2 );

            kl.response = kl.lineLength / std::max( im.cols, im.rows );
            cv::LineIterator li( im, cv::Point2f( fld_lines[i][0], fld_lines[i][1] ), cv::Point2f( fld_lines[i][2], fld_lines[i][3] ) );
            kl.numOfPixels = li.count;
            line.push_back( kl );
        }
        // //检测线特征
        // lsd_->detect(im,line,1,1);
            
        // int lsdNFeatures = nLineFeatures_;
        // struct sort_lines_by_response
        // {
        //     inline bool operator()(const cv::line_descriptor::KeyLine& a, const cv::line_descriptor::KeyLine& b){
        //         return ( a.response > b.response );
        //     }
        // };

        // // filter lines
        // if(line.size()>lsdNFeatures)
        // {
        //     sort(line.begin(), line.end(), sort_lines_by_response());
        //     line.resize(lsdNFeatures);
        //     for(unsigned int i=0; i<lsdNFeatures; i++)
        //         line[i].class_id = i;
        // }

        // int total, index;
        // int nLSDFeature=lsdNFeatures;
        // if(line.size()>nLSDFeature)
        // {
        //     total = nLSDFeature;
        //     index = nLSDFeature;

        // }else{
        //     total = line.size();
        //     index = line.size();
        // }
        // int min_line_length=nminLineLength_;
        // if(line[total-1].lineLength < min_line_length){
        //     for(int i = 0; i<total-1;i++){
        //         if(line[i].lineLength>=min_line_length && line[i+1].lineLength<min_line_length){
        //             index = i;
        //             break;
        //         }
        //     }
        // }

        // line.resize(index + 1);
        // for(unsigned int i=0; i<index + 1; i++){
        //     line[i].class_id = i;
        // }

        // std::cout<<line.size()<<std::endl;
        lbd_->compute( img_clahe, line, lbd_descr );
    }

}