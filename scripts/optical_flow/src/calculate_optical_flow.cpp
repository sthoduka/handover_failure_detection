#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/optflow.hpp>

#include <set>
#include <string>

#include <boost/filesystem.hpp>

namespace bfs = boost::filesystem;

/*
 * This function has been reproduced from the implementation by wanglimin:
 * https://github.com/wanglimin/dense_flow/blob/master/denseFlow.cpp
 */
void convertFlowToImage(const cv::Mat &flow_x, const cv::Mat &flow_y, cv::Mat &img_x, cv::Mat &img_y,
                               double lowerBound, double higherBound) {
#define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
    for (int i = 0; i < flow_x.rows; ++i) {
        for (int j = 0; j < flow_y.cols; ++j) {
            float x = flow_x.at<float>(i,j);
            float y = flow_y.at<float>(i,j);
            img_x.at<uchar>(i,j) = CAST(x, lowerBound, higherBound);
            img_y.at<uchar>(i,j) = CAST(y, lowerBound, higherBound);
        }
    }
#undef CAST
}

int main(int argc, char **argv)
{
    cv::Ptr<cv::optflow::DualTVL1OpticalFlow> alg_tvl1 = cv::optflow::DualTVL1OpticalFlow::create();
    cv::Mat im;

    std::string input_file;
    std::string output_directory;
    if (argc < 3)
    {
        std::cout << "specify input file and output image directory" << std::endl;
        return 1;
    }
    input_file = std::string(argv[1]);
    output_directory = std::string(argv[2]);

    if (!bfs::exists(input_file))
    {
        std::cerr << "file does not exist: " << input_file << std::endl;
        return 1;
    }
    std::cout << input_file << std::endl;

    if (!bfs::exists(output_directory) || !bfs::is_directory(output_directory))
    {
        std::cerr << "Creating " << output_directory << std::endl;
        bfs::create_directory(output_directory);
    }

    cv::VideoCapture cap(input_file);
    if (!cap.isOpened())
    {
        std::cout << "Couldn't open " << input_file << std::endl;
        return 1;
    }
    cap >> im;

    int frame_number = 0;
    cv::Mat output;

    // calculate optical flow using every Nth frame
    int divisor = 1;

    cv::Mat previous_image;
    im.copyTo(previous_image);

    cv::cvtColor(previous_image, previous_image, cv::COLOR_BGR2GRAY);
    // Equalize histogram so that the optical flow calculation is more robust to lighting changes
    cv::equalizeHist(previous_image, previous_image);

    cv::Mat flow;

    // upper bound for optical flow magnitude
    // i.e. magnitudes >= 15 are mapped to 1 and <= -15 are mapped to 0
    int bound = 15;


    while(true)
    {
        frame_number++;
        if (frame_number % divisor !=0) continue;
        std::cout << frame_number << std::endl;

        cap >> im;

        if (im.empty())
        {
            std::cout << "done" << std::endl;
            break;
        }

        cv::Mat curr_gray;
        cv::cvtColor(im, curr_gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(curr_gray, curr_gray);

        alg_tvl1->calc(curr_gray, previous_image, flow);
        cv::Mat flow_parts[2];
        cv::split(flow, flow_parts);
        cv::Mat flow_img_x(flow_parts[0].size(), CV_8UC1);
        cv::Mat flow_img_y(flow_parts[1].size(), CV_8UC1);

        convertFlowToImage(flow_parts[0], flow_parts[1], flow_img_x, flow_img_y,
                          -bound, bound);

        int frame_id = frame_number;
        char tmp[256];
        sprintf(tmp, "frame%04d_x.jpg", frame_id);
        std::string path = output_directory + "/" + tmp;
        cv::imwrite(path, flow_img_x);
        sprintf(tmp, "frame%04d_y.jpg", frame_id);
        path = output_directory + "/" + tmp;
        cv::imwrite(path, flow_img_y);

        curr_gray.copyTo(previous_image);

    }
    return 0;
}
