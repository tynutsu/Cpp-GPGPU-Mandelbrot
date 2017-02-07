#include <cmath>
#include <limits>

#include "MandelbrotSet.h"

using uchar = unsigned char;


#ifndef TOTAL_SHADES
#define TOTAL_SHADES 16
#endif

const uchar maxit = std::numeric_limits<uchar>::max();

MandelbrotSet *set;

struct Complex { double x, y; };

__constant__ Pixel shades[TOTAL_SHADES] =
{ { 66,30,15 },{ 25,7,26 },{ 9,1,47 },{ 4,4,73 },{ 0,7,100 },
{ 12,44,138 },{ 24,82,177 },{ 57,125,209 },{ 134,181,229 },{ 211,236,248 },
{ 241,233,191 },{ 248,201,95 },{ 255,170,0 },{ 204,128,0 },{ 153,87,0 },
{ 106,52,3 } };

__global__ void calc_mandel(Pixel  *img_data, const int width, const int height, const double scale)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int index = row * width + col;
	float x0 = ((float)col / width) * 3.5f - 2.5f;
	float y0 = ((float)row / height) * 3.5f - 1.75f;

	float x = 0.0f;
	float y = 0.0f;
	int iter = 0;
	float xtemp;
	while ((x * x + y * y <= 4.0f) && (iter < maxit))
	{
		xtemp = x * x - y * y + x0;
		y = 2 * x * y + y0;
		x = xtemp;
		iter++;
	}
	if (iter == maxit || iter == 0) {
		img_data[index].r = 0; img_data[index].g = 0; img_data[index].b = 0;
	}
	else {
		img_data[index] = shades[iter % TOTAL_SHADES];
	}	
}

 
void process(MandelbrotSet* set, double scale) {
	dim3 block_size(16, 16);
	int w = set->getWidth();
	int h = set->getHeight();
	dim3 grid_size(w / block_size.x, h / block_size.y);	
	calc_mandel << <grid_size, block_size >> >(set->getDeviceReference(), w, h, scale);
	set->saveAs("testOutput.ppm");
}

int main(int argc, char *argv[])
{
  const int width  = (argc > 1) ? std::atoi(argv[1]) : 4096;
  const int height = (argc > 2) ? std::atoi(argv[2]) : 4096;
  const double scale = 1. / (width / 4);
  set = new MandelbrotSet(width, height);
  for (int i = 0; i < 5; i++) {
	  cout << "Attempt [" << i << "] " << endl;
	  measure(process, set, scale);
  }
  delete set;
  //std::cin.get();
  return 0;
}
