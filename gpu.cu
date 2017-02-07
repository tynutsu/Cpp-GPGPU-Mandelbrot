#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include "MandelbrotSet.h"

using uchar = unsigned char;
using namespace std;
using namespace chrono;

const uchar maxit = std::numeric_limits<uchar>::max();

void check(cudaError_t err, string message, string errorMessage = "");

void check(cudaError err, string message, string errorMessage) {
	if (err != 0) {
		std::cout << errorMessage << ": ";
		std::cout << cudaGetErrorString(err) << __FILE__ << __LINE__ << endl;
	}
	else {
		std::cout << message << endl;
	}
}

struct Complex { double x, y; };
Pixel *setOnHost;			// the Mandelbrot set on the host RAM
Pixel *setOnDevice;			// the Mandelbrot set on the device RAM

void screen_dump(const int width, const int height)
{
  FILE *fp = fopen("gpuOutput.ppm", "w");
  fprintf(fp, "P6\n%d %d\n255\n", width, height);
  for (int i = height - 1; i >= 0; i--) {
	  fwrite(setOnHost + i * width, 1, width * sizeof(Pixel), fp);
  }
  fclose(fp);
}
 
const uchar num_shades = 16;

__constant__ Pixel shades[num_shades] =
{ { 66,30,15 },{ 25,7,26 },{ 9,1,47 },{ 4,4,73 },{ 0,7,100 },
{ 12,44,138 },{ 24,82,177 },{ 57,125,209 },{ 134,181,229 },{ 211,236,248 },
{ 241,233,191 },{ 248,201,95 },{ 255,170,0 },{ 204,128,0 },{ 153,87,0 },
{ 106,52,3 } };

__global__ void calc_mandel(Pixel  *img_data, const int width, const int height, const double scale)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = row * width + col;
	float x0 = ((float)col / width) * 3.5f - 2.5f;
	float y0 = ((float)row / height) * 3.5f - 1.75f;

	float x = 0.0f;
	float y = 0.0f;
	int iter = 0;
	float xtemp;
	while ((x * x + y * y <= 4.0f) && (iter < maxit))
	{
		xtemp = x * x - y * y + x0;
		y = 2.0f * x * y + y0;
		x = xtemp;
		iter++;
	}
	if (iter == maxit || iter == 0) {
		img_data[idx].r = 0; img_data[idx].g = 0; img_data[idx].b = 0;
	}
	else {
		img_data[idx] = shades[iter % num_shades];
	}
	
}

 
void run(int width, int height, double scale) {
	setOnHost = new Pixel[width * height];

	check(cudaMalloc(&setOnDevice, width * height* sizeof(Pixel)),
		"Allocating memory for set", 
		"Could not allocate memory for setOnDevice");
	
	dim3 block_size(16, 16);
	dim3 grid_size(width / block_size.x, height / block_size.y);
	calc_mandel << <grid_size, block_size >> >(setOnDevice, width, height, scale);
	
	check(cudaPeekAtLastError(), 
		"Successfully generated pixels for the set",
		"Something went wrong when executing the kernel calc_mandel");
	
	check(cudaDeviceSynchronize(),
		"Successfully synchronized all processing units",
		"Error occured when cudaDeviceSynchronize()");
	
	check(cudaMemcpy(setOnHost, setOnDevice, width * height * sizeof(Pixel), cudaMemcpyDeviceToHost),
		"Successfully transferred the set to the host",
		"Could not copy the set from device to host");
	
	check(cudaFree(setOnDevice),
		"Successfully cleaned memory on device for setOnDevice",
		"Error occurred when trying to free memory on the GPU for setOnDevice");
	
	screen_dump(width, height);
}

int main(int argc, char *argv[])
{
  const int width  = (argc > 1) ? std::atoi(argv[1]) : 4096;
  const int height = (argc > 2) ? std::atoi(argv[2]) : 4096;
  const double scale = 1. / (width / 4);
  cout << endl << endl << endl << "Scale: " << scale << endl << endl << endl;
  for (int i = 0; i < 5; i++) {
	  auto start = high_resolution_clock::now();
	  run(width, height, scale);
	  auto end = high_resolution_clock::now();
	  auto elapsed = duration_cast<milliseconds> (end - start);
	  std::cout << "Execution time: " << elapsed.count() << " miliseconds " << i << std::endl;
  }
  std::cin.get();
  delete setOnHost;
  return 0;
}
