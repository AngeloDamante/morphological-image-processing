/*
 * Probe.h
 * Definition probe class for structuring element.
 *
 * @Author: AngeloDamante
 * @mail: angelo.damante16@gmail.com
 * @GitHub: https://github.com/AngeloDamante
 */

#ifndef MM_PROBE_H
#define MM_PROBE_H

class Probe {
public:
explicit Probe() = delete;
explicit Probe(int xRadius, int yRadius);
explicit Probe(int radius, float *data);
explicit Probe(int xRadius, int yRadius, float *data);
virtual ~Probe() {
	delete[] this->data;
}

void printMask() const;

// setter methods
void setData(float *data);

// getter methods
inline int getXRadius() const {
	return this->xRadius;
}
inline int getYRadius() const {
	return this->yRadius;
}
inline int getWidth() const {
	return this->width;
}
inline int getHeight() const {
	return this->height;
}
inline int getSize() const {
	return height * width;
}
inline float *getData() const {
	return this->data;
}

protected:
int xRadius;
int yRadius;
int width;
int height;
float *data;
};

class Square : public Probe {
public:
explicit Square(int radius);
};

class Rectangle : public Probe {
public:
explicit Rectangle(int xRadius, int yRadius);
};

#endif // MM_PROBE_H
