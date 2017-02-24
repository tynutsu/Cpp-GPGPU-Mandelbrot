#pragma once
#include "Util.h"

class MandelbrotSet {
private:
	Pixel *setOnHost;			// the Mandelbrot set data on the host RAM
	Pixel *setOnDevice;			// the Mandelbrot set on data the device RAM
	Complex complexNumber;		// the variable complex number which belongs or not to the MandelbrotSet

	int width = 4096;
	int height = 4096;

	void init();

public:
	MandelbrotSet();
	MandelbrotSet(int width, int height, Complex number);
	~MandelbrotSet();

	Pixel *getHostReference() { return setOnHost; }
	Pixel *getDeviceReference() { return setOnDevice; }
	int getWidth() { return width; }
	int getHeight() { return height; }
	Complex getComplex() { return complexNumber; }

	void fetch();

	void saveAs(std::string fileName) {
		std::ofstream file(fileName.c_str());
		file << "P6\n" << width << " " << height << "\n255\n";
		file.write((char*) setOnHost, height * width * sizeof(Pixel));
	}
};

