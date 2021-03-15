/*
 * MorphologicalOperator.h
 * Definition morpological operators.
 *      - Dilatation (D)
 *      - Erosion (E)
 *      - Opening = D(E(Image, probe), probe)
 *      - Closing = E(D(Image, probe), probe)
 *
 *  Created on: 23/feb/2021
 *      Author: AngeloDamante
 */

#ifndef MM_MATHEMATICAL_MORPHOLOGY_H
#define MM_MATHEMATICAL_MORPHOLOGY_H

#include "Image.h"
#include "Probe.h"

enum mmOp { DILATATION, EROSION, OPENING, CLOSING };

namespace operation {
    class MathematicalMorphology {
    public:
      MathematicalMorphology() = delete;
      static inline Image *mm(Image *image, Probe *probe, mmOp mmOp) {
        switch (mmOp) {
        case (EROSION):
          return erosion(image, probe);
        case (DILATATION):
          return dilatation(image, probe);
        case (OPENING):
          return opening(image, probe);
        case (CLOSING):
          return closing(image, probe);
          break;
        }
      }

    protected:
      static Image *erosion(Image *image, Probe *probe);
      static Image *dilatation(Image *image, Probe *probe);
      static Image *opening(Image *image, Probe *probe);
      static Image *closing(Image *image, Probe *probe);

    private:
      static float *__process(Image *image, Probe *probe, mmOp basicOp);
    };

    namespace utils {
        float max(float *src, int length);
        float min(float *src, int length);
        void initilize(float *src, int length);
    } // namespace utils
} // namespace operation

#endif // MM_MATHEMATICAL_MORPHOLOGY_H
