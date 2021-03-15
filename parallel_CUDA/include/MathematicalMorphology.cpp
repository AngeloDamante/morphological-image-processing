#include "MathematicalMorphology.h"

/*** MathematicalMorphology ***/
Image *operation::MathematicalMorphology::erosion(Image *image, Probe *probe) {
  float *outData = __process(image, probe, EROSION);
  Image *out = new Image(image->getHeight(), image->getWidth(), outData, bw);

  return out;
}

Image *operation::MathematicalMorphology::dilatation(Image *image,
                                                     Probe *probe) {
  float *outData = __process(image, probe, DILATATION);
  Image *out = new Image(image->getHeight(), image->getWidth(), outData, bw);

  return out;
}

Image *operation::MathematicalMorphology::opening(Image *image, Probe *probe) {
  Image *imageEroded = erosion(image, probe);
  Image *out = dilatation(imageEroded, probe);

  return out;
}

Image *operation::MathematicalMorphology::closing(Image *image, Probe *probe) {
  Image *imageDilatated = dilatation(image, probe);
  Image *out = erosion(imageDilatated, probe);

  return out;
}

float *operation::MathematicalMorphology::__process(Image *image, Probe *probe,
                                                    mmOp basicOp) {
  float *imageData = image->getData();
  float *probeData = probe->getData();

  int imgH = image->getHeight();
  int imgW = image->getWidth();
  int prbH = probe->getHeight();
  int prbW = probe->getWidth();

  float *outData = new float[imgH * imgW];
  float *neighborhood = new float[prbH * prbW];

  for (int rowImg = 0; rowImg < imgH; rowImg++) {
    for (int colImg = 0; colImg < imgW; colImg++) {
      int numNeighboor = 0;
      for (int rowPrb = 0; rowPrb < prbH; rowPrb++) {
        for (int colPrb = 0; colPrb < prbW; colPrb++) {
          int x = colImg - probe->getXRadius() + colPrb;
          int y = rowImg - probe->getYRadius() + rowPrb;
          if (x > -1 && x < imgW && y > -1 && y < imgH &&
              probeData[rowPrb * prbW + colPrb] > 0) {
            neighborhood[numNeighboor] = imageData[y * imgW + x];
            numNeighboor++;
          }
        }
      }
      if (basicOp == EROSION)
        outData[rowImg * imgW + colImg] =
            utils::min(neighborhood, numNeighboor);
      if (basicOp == DILATATION)
        outData[rowImg * imgW + colImg] =
            utils::max(neighborhood, numNeighboor);
    }
  }

  delete[] neighborhood;
  return outData;
}

/*** utils ***/
float operation::utils::max(float *src, int length) {
  float max = src[0];
  for (int i = 0; i < length; i++) {
    if (max < src[i])
      max = src[i];
  }
  return max;
}

float operation::utils::min(float *src, int length) {
  float min = src[0];
  for (int i = 0; i < length; i++) {
    if (min > src[i])
      min = src[i];
  }
  return min;
}

void operation::utils::initilize(float *src, int length) {
  for (int i = 0; i < length; i++) {
    src[i] = 0;
  }
}
