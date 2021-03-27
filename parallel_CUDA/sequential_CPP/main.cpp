/*
 * main.cpp
 * Main that build sequential version for MM operations.
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
#include "MathematicalMorphology.h"
#include "Probe.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

const std::string inputPath = "/images";
const std::string outputPath = "/results_CUDA";

int main(int argc, char const *argv[]) {
  Image *inputImg, *outputImg;
  Probe *probe = new Square(1); // mask 3x3
  std::chrono::high_resolution_clock::time_point start, end;
  std::chrono::duration<double> span;
  std::ofstream resultsFile;
  std::string pathImg;

  resultsFile.open(outputPath + "/sequential_timings.csv");
  for (const auto &img : fs::recursive_directory_iterator(inputPath)) {
    pathImg = img.path();

    // Verify that is image.png and not a simple dir_name
    if(pathImg.rfind('.') != std::string::npos){

        inputImg = new Image(pathImg);
        inputImg->rgb2bw(); // extract only first channel

        resultsFile << pathImg << "\n";
        for(const auto& op : operation::MM){
            std::cout << op.second << " operation in ...";

            start = std::chrono::high_resolution_clock::now();
            outputImg = operation::MathematicalMorphology::mm(inputImg, probe, op.first);
            end = std::chrono::high_resolution_clock::now();
            span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
            //outputImg->saveImg(...);

            std::cout << span.count() <<std::endl;
            resultsFile << op.second << ";" << span.count() << "\n";
        }

    }else{

        std::cout << "\n" << "PROCESSING------->" << pathImg << std::endl;
        resultsFile << pathImg << "\n";

    }

  }

  delete inputImg, probe;
  resultsFile.close();

  printf("\n *** Completed! *** \n");
  return 0;
}
