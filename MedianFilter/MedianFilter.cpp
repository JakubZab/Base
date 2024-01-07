#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <omp.h>
#include <chrono>

using namespace cv;

#define NUM_THREADS 4

#define WINDOW_WIDTH 5
#define WINDOW_HEIGHT 5
#define EDGE_X ((int) (WINDOW_WIDTH/2))
#define EDGE_Y ((int) (WINDOW_HEIGHT/2))
#define MEDIAN_INDEX ((int)(WINDOW_WIDTH * WINDOW_HEIGHT / 2))

void blur(cv::Mat* src, cv::Mat* dst) {
    #pragma omp parallel
    {
        printf("My ID is %d\n", omp_get_thread_num());
        #pragma omp for
        for (int c = EDGE_X; c < src->cols - EDGE_X; c++) {
            for (int r = EDGE_Y; r < src->rows - EDGE_Y; r++) {
                std::vector<int> windowValues;
                for (int window_c = 0; window_c < WINDOW_WIDTH; window_c++) {
                    for (int window_r = 0; window_r < WINDOW_HEIGHT; window_r++) {
                        windowValues.push_back(src->at<unsigned char>(Point(r + window_r - EDGE_Y,
                            c + window_c - EDGE_X)));
                    }
                }
                sort(windowValues.begin(), windowValues.end());
                dst->at<unsigned char>(c, r) = windowValues.at(MEDIAN_INDEX);
            }
        }
    };
}

int main()
{
    cv::Mat img = cv::imread("F:\\Nowy folder\\II semestr\\Procesory i Architektury Systemów Komputerowych\\Projekt\\OMP\\Base\\MedianFilter\\lenna.png");
    cv::Mat grayscaled;
    cv::Mat processed;

    cv::cvtColor(img, grayscaled, CV_BGR2GRAY);

    std::cout << "Starting single thread run\n";
    omp_set_num_threads(1);
    processed = grayscaled.clone();
    auto t1 = std::chrono::high_resolution_clock::now();
    blur(&grayscaled, &processed);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Single thread execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms\n";

    std::cout << "Starting " << NUM_THREADS << " thread run\n";
    omp_set_num_threads(NUM_THREADS);
    processed = grayscaled.clone();
    t1 = std::chrono::high_resolution_clock::now();
    blur(&grayscaled, &processed);
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << NUM_THREADS << " thread execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms\n";


    namedWindow("First OpenCV Application", WINDOW_AUTOSIZE);
    cv::imshow("First OpenCV Application", grayscaled);
    cv::moveWindow("First OpenCV Application", 0, 45);
    cv::waitKey(0);
    cv::imshow("First OpenCV Application", processed);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}