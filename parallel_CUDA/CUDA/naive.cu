/*
 * naive.cu
 * Main that build parallel version for MM operations in naive solution.
 *
 * Images analyzed:
 *      - bin.png (640x480)
 *      - mandel.png (1280x960)
 *      - simpson.png (1920x1200)
 *      - lion.png (2560x1440)
 *      - homer.png (3840x2160)
 *
 * @Author: AngeloDamante
 * @mail: angelo.damante16@gmail.com
 * @GitHub: https://github.com/AngeloDamante
 *
 */

#include "Image.h"
#include "Probe.h"
#include "MM.cuh"
#include <cstring>
#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>

const std::string inputPath = "/images/examples/";
const std::string outputPath = "/images/gpuResults/";
const std::string testPath = "/images/tests/";

const std::vector<std::string> operationsName = {"DILATATION", "EROSION",
                                                 "OPENING", "CLOSING"};

const std::vector<MMop> morphOperations = {DILATATION, EROSION, OPENING,
                                           CLOSING};

const std::vector<std::string> images = {"bin640x480", "mandel1280x960",
                                         "simpson1920x1200", "lion2560x1440",
                                         "homer3840x2160"};

int main(int argc, char const *argv[]) {

    Image *inputImg, *outputImg;
    Probe *probe = new Square(MASK_RADIUS);
    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> span;
    std::ofstream resultsFile;

    resultsFile.open("/parallelVersion/CUDA/naive_timing.csv");

    for ( const auto &img : images) {
        inputImg = new Image(inputPath + img + ".png");
        inputImg->rgb2bw();

        std::cout << "\n" << img << " processing:" << std::endl;

        resultsFile << img << "\n";

        // Every operation in naive version
        for (const auto &op : morphOperations) {
            std::cout << operationsName[op] << "...";

            start = std::chrono::high_resolution_clock::now();
            outputImg = mm(inputImg, probe, op, NAIVE);
            end = std::chrono::high_resolution_clock::now();
            span = end - start;
            span = std::chrono::duration_cast<std::chrono::duration<double>>(span);

            // outputImg->saveImg(testPath + img + "__" + operationsName[op] + ".png");
            outputImg->saveImg(outputPath + img + "__" + operationsName[op] + ".png");

            std::cout << span.count() << std::endl;
            resultsFile << operationsName[op] << ";" << span.count() << "\n";
        }

        delete inputImg;
    }
    delete probe;

    resultsFile.close();

    printf("\n *** Completed! *** \n");
    return 0;
}
