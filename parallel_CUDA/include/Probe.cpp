#include "Probe.h"
#include <cstdio>

Probe::Probe(int xRadius, int yRadius) {
	this->xRadius = xRadius;
	this->yRadius = yRadius;
	this->width = 2 * xRadius + 1;
	this->height = 2 * yRadius + 1;

	this->data = new float[height * width];
	for (int i = 0; i < width * height; i++)
		this->data[i] = 0;
}

Probe::Probe(int radius, float *data) {
	Probe(radius, radius);
	delete[] this->data;
	this->data = data;
}

Probe::Probe(int xRadius, int yRadius, float *data) {
	Probe(xRadius, yRadius);
	delete[] this->data;
	this->data = data;
}

void Probe::setData(float *data) {
	delete[] this->data;
	this->data = data;
}

void Probe::printMask() const {
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++)
			printf("%f   ", data[row * width + col]);
		printf("\n");
	}
}

Square::Square(int radius) : Probe(radius, radius) {
	for (int i = 0; i < width * height; i++)
		this->data[i] = 1;
}

Rectangle::Rectangle(int xRadius, int yRadius) : Probe(xRadius, yRadius) {
	for (int i = 0; i < width * height; i++)
		this->data[i] = 1;
}
