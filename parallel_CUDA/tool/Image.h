/*
 * Image.h
 *
 *  Created on: 23/feb/2021
 *      Author: AngeloDamante
 */

#include "CImg.h"
#include <cstring>

using namespace cimg_library;

class Image {

public:
  Image(int height, int width, int numChannels)
      : width(width), height(height), numChannels(numChannels);

  Image(std::string path);

  virtual ~Image() { delete[] data; }

private:
  int height;
  int width;
  int numChannels;
  float *data;
};
