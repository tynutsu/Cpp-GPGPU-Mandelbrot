#include "MandelbrotSet.h"

MandelbrotSet::MandelbrotSet()
{
	init();
}

MandelbrotSet::MandelbrotSet(int width, int height) {
	this->width = width;
	this->height = height;
	init();
}

MandelbrotSet::~MandelbrotSet()
{
	check(cudaFree(setOnDevice),
		"Successfully cleaned memory on device for setOnDevice",
		"Error occurred when trying to free memory on the GPU for setOnDevice");
}

void MandelbrotSet::init() {
	setOnHost = new Pixel[width * height];

	check(cudaMalloc(&setOnDevice, width * height* sizeof(Pixel)),
		"Allocating memory for set",
		"Could not allocate memory for setOnDevice");
}

void MandelbrotSet::fetch() {
	check(cudaDeviceSynchronize(),
		"Successfully synchronized all processing units",
		"Error occured when cudaDeviceSynchronize()");

	check(cudaMemcpy(setOnHost, setOnDevice, width * height * sizeof(Pixel), cudaMemcpyDeviceToHost),
		"Successfully transferred the set to the host",
		"Could not copy the set from device to host");
}
