/*
 * Image.h
 *
 *  Created on: 23/feb/2021
 *      Author: AngeloDamante
 */

class Image {

public:
  inline Image(int height, int width, int numChannels, float* data)
      : width(width), height(height), numChannels(numChannels), data(data) {}

  virtual ~Image() { delete[] data; }

  Image(const char* path);

  inline int getHeight() { return this->height; }
  inline int getWidth() { return this->width; }
  inline int getNumChannels() { return this->numChannels; }
  inline float* getData() { return this->data; }

private:
  int height;
  int width;
  int numChannels;
  float *data;

};
