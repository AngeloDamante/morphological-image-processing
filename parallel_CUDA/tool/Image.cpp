#include "CImg.h"
#include "Image.h"

using namespace cimg_library;

Image::Image(const char *pathImg) {
  CImg<float> image(pathImg);
  this->height = image.height();
  this->width = image.width();
  this->numChannels = image.spectrum();
  this->data = image.data();
}

void Image::rgb2bn() {
  if (this->numChannels > 1) {
    float *bufferBn = (float *)malloc(sizeof(float) * height * width);
    for (int i = 0; i < height * width; i++)
      bufferBn[i] = this->data[i];
    this->data = bufferBn;
    this->numChannels = 1;
  }
}

float *Image::getBinaryBufferBn() {
  /*0.0 = black; 1.0 = white;*/
  float *binaryData = nullptr;
  if (this->numChannels == 1) {
    binaryData = (float *)malloc(sizeof(float) * height * width);
    for (int i = 0; i < height * width; i++)
      binaryData[i] = this->data[i] / 255;
  }
  return binaryData;
}

void Image::saveImg(const char *pathSave) {
  CImg<float> image(data, width, height, 1, numChannels);
  image.save(pathSave);
}

void Image::setData(float *data, bool binary) {
  if (binary) {
    for (int i = 0; i < height * width; i++)
      data[i] = data[i] * 255;
  }
  this->data = data;
}
