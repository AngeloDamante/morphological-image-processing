/*
 * MorphologicalOperator.h
 * Definition morpological operators.
 *      - Dilatation (D)
 *      - Erosion (E)
 *      - Opening = D(E(Image, probe), probe)
 *      - Closing = E(D(Image, probe), probe)
 *
 * @Author: AngeloDamante
 * @mail: angelo.damante16@gmail.com
 * @GitHub: https://github.com/AngeloDamante
 */

#ifndef MM_MATHEMATICAL_MORPHOLOGY_H
#define MM_MATHEMATICAL_MORPHOLOGY_H

#include "Image.h"
#include "Probe.h"
#include <cstring>
#include <map>

enum mmOp { DILATATION, EROSION, OPENING, CLOSING };

namespace operation {

const std::map<mmOp, std::string> MM = {{DILATATION, "Dilatation"},
	{EROSION, "Erosion"},
	{CLOSING, "Closing"},
	{OPENING, "Opening"}

};

class MathematicalMorphology {
public:
MathematicalMorphology() = delete;

/// Interface for choosing the MM operation
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
/**
 * MM operation: erosion | dilatation | opening | closing
 *
 * @param Image* image: input image to processing.
 * @param Probe* probe: structuring element.
 *
 * @return Image* image processed.
 */
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

}     // namespace utils

} // namespace operation

#endif // MM_MATHEMATICAL_MORPHOLOGY_H
