/*
 * sharedOpt.cu
 * Main that build parallel version for MM operations in optimized solution.
 *
 * This solution use shared memory and tailing techniques.
 * The probe element is stored on constant memory.
 *
 *
 * @Author: AngeloDamante
 * @mail: angelo.damante16@gmail.com
 * @GitHub: https://github.com/AngeloDamante
*/

#include "Image.h"
#include "Probe.h"
#include "MM.cuh"
#include <cstring>
#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>

#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

const std::string inputPath = "/images";
const std::string outputPath = "/results_CUDA";

int main(int argc, char const *argv[]) {

    Image *inputImg, *outputImg;
    Probe *probe = new Square(MASK_RADIUS);
    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> span;
    std::ofstream resultsFile;
    std::string pathImg;

    resultsFile.open(outputPath + "/optimized_timing.csv");
    for(const auto& img : fs::recursive_directory_iterator(inputPath)){
        pathImg = img.path();

        // Verify that is image.png and not a simple dir_name
        if(pathImg.rfind('.') != std::string::npos){

            inputImg = new Image(pathImg);
            inputImg -> rgb2bw();

            resultsFile << pathImg << "\n";
            for(const auto& op : MMoperations){
                std::cout << op.second << " operation in ...";

                start = std::chrono::high_resolution_clock::now();
                outputImg = mm(inputImg, probe, op.first, SHAREDOPT);
                end = std::chrono::high_resolution_clock::now();
                span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
                //outputImg->saveImg(...);

                std::cout << span.count() <<std::endl;
                resultsFile << op.second << ";" << span.count() << "\n";
            }

            delete inputImg;
        }else{
            std::cout << "\n" << "PROCESSING------->" << pathImg << std::endl;
            resultsFile << pathImg << "\n";
        }

    }

    delete probe;
    resultsFile.close();

    printf("\n *** Completed! *** \n");
    return 0;
}
