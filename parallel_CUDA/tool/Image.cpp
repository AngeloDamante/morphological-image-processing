#include "Image.h"
#include <cstdio>
#include <iostream>

Image::Image(const std::string pathImg) {
  cimg_library::CImg<float> image(pathImg.c_str());
  this->height = image.height();
  this->width = image.width();
  this->numChannels = image.spectrum();
  this->state = (numChannels > 1) ? rgb : bw;

  this->data = new float[image.size()];
  for (int i = 0; i < image.size(); i++)
    this->data[i] = image.data()[i];
}

void Image::saveImg(const std::string pathSave) const {
  cimg_library::CImg<float> image(data, width, height, 1, numChannels);
  image.save(pathSave.c_str());
}

void Image::setState(stateImg state) {
  this->state = state;
  this->numChannels = (state != rgb) ? 1 : 3;
}

void Image::rgb2bw() {
  if (state == rgb) {
    float *bufferBn = new float[height * width];
    for (int i = 0; i < height * width; i++) {
      bufferBn[i] = (this->data)[i];
    }

    delete[] this->data;
    this->setData(bufferBn);
    this->setState(bw);
  }
}

void Image::bw2gray() {
  if (state == bw) {
    for (int i = 0; i < height * width; i++) {
      this->data[i] = this->data[i] / 255;
    }
    this->setState(grayscale);
  }
}

void Image::gray2bw() {
  if (state == grayscale) {
    for (int i = 0; i < height * width; i++) {
      this->data[i] = this->data[i] * 255;
    }
    this->setState(bw);
  }
}

void Image::rgb2gray() {
  this->rgb2bw();
  this->bw2gray();
}
