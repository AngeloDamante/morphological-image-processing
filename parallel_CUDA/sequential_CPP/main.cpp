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

const std::string inputPath = "/images/examples/";
const std::string outputPath = "/images/seqResults/";

const std::vector<std::string> operationsName = {"DILATATION", "EROSION",
                                                 "OPENING", "CLOSING"};

const std::vector<mmOp> morphOperations = {DILATATION, EROSION, OPENING,
                                           CLOSING};

const std::vector<std::string> images = {"bin640x480", "mandel1280x960",
                                         "simpson1920x1200", "lion2560x1440",
                                         "homer3840x2160"};

int main(int argc, char const *argv[]) {
  Image *inputImg, *outputImg;
  Probe *probe = new Square(2); // mask 3x3
  std::chrono::high_resolution_clock::time_point start, end;
  std::chrono::duration<double> span;
  std::ofstream resultsFile;

  resultsFile.open("/parallelVersion/sequential_CPP/sequential_timings.csv");
  for (const auto &img : images) {
    inputImg = new Image(inputPath + img + ".png");
    inputImg->rgb2bw(); // extract only first channel

    std::cout << "\n" << img << " processing:" << std::endl;

    resultsFile << img << "\n";
    for (const auto &mm : morphOperations) {
      std::cout << operationsName[mm] << "...";

      start = std::chrono::high_resolution_clock::now();
      outputImg = operation::MathematicalMorphology::mm(inputImg, probe, mm);
      end = std::chrono::high_resolution_clock::now();
      span = end - start;
      span = std::chrono::duration_cast<std::chrono::duration<double>>(span);

      outputImg->saveImg(outputPath + img + "__" + operationsName[mm] + ".png");

      std::cout << span.count() << std::endl;
      resultsFile << operationsName[mm] << ";" << span.count() << "\n";
    }

    delete inputImg;
  }
  delete probe;

  resultsFile.close();

  printf("\n *** Completed! *** \n");
  return 0;
}
