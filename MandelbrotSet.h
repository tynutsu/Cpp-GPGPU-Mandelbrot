#pragma once
#include "Util.h"

class MandelbrotSet
{
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

	inline void saveAs(std::string fileName) {
		FILE *fp = fopen(fileName.c_str(), "w");
		fprintf(fp, "P6\n%d %d\n255\n", width, height);
		for (int i = height - 1; i >= 0; i--) {
			fwrite(setOnHost + i * width, 1, width * sizeof(Pixel), fp);
		}
		fclose(fp);
	}
};

