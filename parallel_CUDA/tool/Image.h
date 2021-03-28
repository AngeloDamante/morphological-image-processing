/*
 * Image.h
 * A simple wrapper class for Image type.
 *
 * @Author: AngeloDamante
 * @mail: angelo.damante16@gmail.com
 * @GitHub: https://github.com/AngeloDamante
 */

#ifndef MM_IMAGE_H
#define MM_IMAGE_H

#define cimg_display 0
#define cimg_use_png 1
#include "CImg.h"
#include <cstring>
#include <iostream>

enum stateImg { rgb, grayscale, bw };

class Image {
public:
  inline explicit Image(int height, int width, float *data, stateImg state)
      : width(width), height(height), data(data), state(state) {
    this->numChannels = (state != rgb) ? 1 : 3;
  }

  virtual ~Image() { delete[] data; }

  void saveImg(const std::string pathSave) const;

  // @overload constructor
  explicit Image(const std::string pathImg);

  // conversions
  void rgb2bw();
  void bw2gray();
  void gray2bw();
  void rgb2gray();

  // setter methods
  inline void setData(float *data) { this->data = data; }
  void setState(stateImg state);

  // getter methods
  inline int getHeight() const { return this->height; }
  inline int getWidth() const { return this->width; }
  inline int getNumChannels() const { return this->numChannels; }
  inline stateImg getState() const { return this->state; }
  inline int getSize() const { return (height * width * numChannels); }
  inline float *getData() const { return this->data; }

private:
  int height;
  int width;
  int numChannels;
  float *data;
  stateImg state;
};

#endif // MM_IMAGE_H
