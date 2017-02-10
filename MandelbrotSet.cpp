#include "MandelbrotSet.h"
#define LOG false

MandelbrotSet::MandelbrotSet()
{
	init();
}

MandelbrotSet::MandelbrotSet(int width, int height, Complex number)
{
	this->width = width;
	this->height = height;
	this->complexNumber = number;
	init();
}

MandelbrotSet::~MandelbrotSet()
{
	delete setOnHost;
	check(cudaFree(setOnDevice),
		"Freed memory on device for setOnDevice",
		"Error occurred when trying to free memory on the GPU for setOnDevice",
		LOG);
}

void MandelbrotSet::init() {
	setOnHost = new Pixel[width * height];

	check(cudaMalloc(&setOnDevice, width * height* sizeof(Pixel)),
		"Allocated memory for set",
		"Could not allocate memory for setOnDevice",
		LOG);
}

void MandelbrotSet::fetch() {
	check(cudaDeviceSynchronize(),
		"Synchronized all processing units",
		"Error occured when cudaDeviceSynchronize()",
		LOG);

	check(cudaMemcpy(setOnHost, setOnDevice, width * height * sizeof(Pixel), cudaMemcpyDeviceToHost),
		"Transferred the set to the host",
		"Could not copy the set from device to host",
		LOG);
}
