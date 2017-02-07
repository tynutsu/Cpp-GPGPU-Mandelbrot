#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

using uchar = unsigned char;
using namespace std::chrono;
const double cx = -.6, cy = 0;
const uchar maxit = std::numeric_limits<uchar>::max();

#define cudaAssertSuccess(ans) { _cudaAssertSuccess((ans), __FILE__, __LINE__); }
inline void _cudaAssertSuccess(cudaError_t code, char *file, int line)
{
	if (code != cudaSuccess) {
		fprintf(stderr, "_cudaAssertSuccess: %s %s %d\n", cudaGetErrorString(code), file, line);
		std::cin.get();
		exit(code);
	}
}

struct Pixel { unsigned char r, g, b; };
struct Complex { double re, im; };

Pixel **row;
Pixel *setOnHost;			// the Mandelbrot set on the host RAM
Pixel *setOnDevice;			// the Mandelbrot set on the device RAM
 
void screen_dump(const int width, const int height)
{
  FILE *fp = fopen("gpuOutput.ppm", "w");
  fprintf(fp, "P6\n%d %d\n255\n", width, height);
  for (int i = height - 1; i >= 0; i--)
    fwrite(row[i], 1, width * sizeof(Pixel), fp);
  fclose(fp);
}
 
void alloc_2d(const int width, const int height)
{
  setOnHost = new Pixel[width * height];
  row = new Pixel*[width];
  row[0] = setOnHost;
  for (int i = 1; i < width; i++)
    row[i] = row[i-1] + height;
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
	if (col >= width || row >= height) return;

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
	alloc_2d(width, height);
	
	cudaAssertSuccess(cudaMalloc(&setOnDevice, width * height* sizeof(Pixel)));
	dim3 block_size(16, 16);
	dim3 grid_size(width / block_size.x, height / block_size.y);
	calc_mandel << <grid_size, block_size >> >(setOnDevice, width, height, scale);
	cudaAssertSuccess(cudaPeekAtLastError());
	cudaAssertSuccess(cudaDeviceSynchronize());
	cudaAssertSuccess(cudaMemcpy(setOnHost, setOnDevice, width * height * sizeof(Pixel), cudaMemcpyDeviceToHost));
	cudaAssertSuccess(cudaFree(setOnDevice));
	screen_dump(width, height);
}

int main(int argc, char *argv[])
{
  const int width  = (argc > 1) ? std::atoi(argv[1]) : 4096;
  const int height = (argc > 2) ? std::atoi(argv[2]) : 4096;
  const double scale = 1./(width/4);
  for (int i = 0; i < 5; i++) {
	  high_resolution_clock::time_point start = high_resolution_clock::now();
	  run(width, height, scale);
	  high_resolution_clock::time_point end = high_resolution_clock::now();
	  auto elapsed = duration_cast<milliseconds> (end - start);
	  std::cout << "Execution time: " << elapsed.count() << " miliseconds " << i << std::endl;
  }
  std::cin.get();
  delete setOnHost;
  delete[] row;
  return 0;
}
