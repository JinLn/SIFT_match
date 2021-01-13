#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <algorithm>
using namespace std;
using namespace cv;


int main()
{
    Mat src1 = imread("./match1.png");
    Mat src2 = imread("./match2.png");
    //特征点提取方法
    cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();

    //特征点提取
    vector<KeyPoint> kp1,kp2;
    sift->detect(src1,kp1);
    sift->detect(src2,kp2);

    //画出特征点
    Mat keypointImage1,key_pointImage2,descriptor1,descriptor2;
    drawKeypoints(src1,kp1,keypointImage1,cv::Scalar::all(-1),DrawMatchesFlags::DEFAULT);

//    imshow("SIFTKP1",keypointImage1);
    drawKeypoints(src2,kp2,key_pointImage2,cv::Scalar(-1),DrawMatchesFlags::DEFAULT);
//    imshow("SIFTKP2",key_pointImage2);

    cv::Mat match_pointL,match_pointR;
    sift->detectAndCompute(src1,cv::Mat(),kp1,match_pointL);
    sift->detectAndCompute(src2,cv::Mat(),kp2,match_pointR);

    if(match_pointL.type()!=CV_32F||match_pointR.type()!=CV_32F)
    {
        match_pointL.convertTo(match_pointL,CV_32F);
        match_pointL.convertTo(match_pointR,CV_32F);
    }

    vector<DMatch> matches;

//    @param descriptorMatcherType Descriptor matcher type. Now the following matcher types are
//    supported:
//    -   `BruteForce` (it uses L2 )
//    -   `BruteForce-L1`
//    -   `BruteForce-Hamming`
//    -   `BruteForce-Hamming(2)`
//    -   `FlannBased`
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
    matcher->match(match_pointL,match_pointR,matches);

    double maxDist = 10;
    for(int i =0;i<match_pointL.rows;i++)
    {
        double dist = matches[i].distance;
        if(dist>maxDist)
            maxDist= dist;
    }

    ////调参褚   0.1越小 越精确  官方推荐0.5 如果确定点 可改变
    double dist=0.5*maxDist;
    //cout<<"dist = "<<dist<<endl;
    vector<DMatch> good_matches,need_matches;
    vector<float>distance;
    //cout << match_pointL.rows<<endl;
    for(int i =0;i<match_pointL.rows;i++)
    {
        if(matches[i].distance<dist)
        {
            good_matches.push_back(matches[i]);
            distance.push_back(matches[i].distance);
        }
    }

    vector<DMatch> best_matches;
    for(int i=0; i<good_matches.size(); i++)
    {
        float dx=kp1[good_matches[i].queryIdx].pt.x - kp2[good_matches[i].trainIdx].pt.x;
        float dy=kp1[good_matches[i].queryIdx].pt.y - kp2[good_matches[i].trainIdx].pt.y;
        float distancePoint =sqrt(dx*dx+dy*dy);
        //cout <<"distancePoint = "<<distancePoint<<endl;
        // 半径为５个像素范围内
        if(distancePoint< 5*5)
            best_matches.push_back(good_matches[i]);
    }
    cout << "matched all points numbers = "<< best_matches.size()<<endl;

    Mat goodimageOutput, bestimageOutput;
    cv::drawMatches(src1,kp1,src2,kp2,good_matches,goodimageOutput);
    cv::drawMatches(src1,kp1,src2,kp2,best_matches,bestimageOutput);
    vector<Point>first_image,second_image;
    for(int i =0;i<need_matches.size();i++)
    {
        first_image.push_back(kp1[need_matches[i].queryIdx].pt);
        second_image.push_back(kp2[need_matches[i].trainIdx].pt);
    }

    cv::imshow("匹配图片good",goodimageOutput);
    cv::imshow("匹配图片best",bestimageOutput);
    waitKey(0);
    return 0;
}
