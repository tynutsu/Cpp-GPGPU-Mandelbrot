#include <cmath>
#include "MandelbrotSet.h"

MandelbrotSet *set;

__constant__ Pixel shades[TOTAL_SHADES] = {
	{ 66,30,15 },
	{ 25,7,26 },
	{ 9,1,47 },
	{ 4,4,73 },
	{ 0,7,100 },
	{ 12,44,138 },
	{ 24,82,177 },
	{ 57,125,209 },
	{ 134,181,229 },
	{ 211,236,248 },
	{ 241,233,191 },
	{ 248,201,95 },
	{ 255,170,0 },
	{ 204,128,0 },
	{ 153,87,0 },
	{ 106,52,3 }
};

__global__ void calc_mandel(Pixel  *data, const int width, const int height, const double scale, const Complex number)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int index = row * width + column;
	float x0 = (column - width/2) * scale - 0.6f;
	float y0 = (row - height/2) * scale + 0.0f;
	float zx = 0.0f, zy = 0.0f, xx = 0.0f, yy = 0.0f;
	unsigned char iter = 0;
	do
	{
		xx = zx * zx;
		yy = zy * zy;
		zy = 2 * zx * zy + y0;
		zx = xx - yy + x0;;
		iter++;
	} while ((xx + yy <= 4.0f) && (iter < maxit));
	if (iter == maxit || iter == 0) {
		data[index].r = 0; data[index].g = 0; data[index].b = 0;
	}
	else {
		data[index] = shades[iter % TOTAL_SHADES];
	}	
}

void process(MandelbrotSet* set, double scale, char* fileName) {
	dim3 block_size(16, 16);
	int w = set->getWidth();
	int h = set->getHeight();
	Complex n = set->getComplex();
	dim3 grid_size(w / block_size.x, h / block_size.y);	
	calc_mandel << <grid_size, block_size >> >(set->getDeviceReference(), w, h, scale, n);
	set->saveAs(fileName);
}

int main(int argc, char *argv[])
{
  const int width  = (argc > 1) ? std::atoi(argv[1]) : 4096;
  const int height = (argc > 2) ? std::atoi(argv[2]) : 4096;
  double real = (argc > 3) ? std::atol(argv[3]) : -0.6f;
  double imaginary = (argc > 4) ? std::atol(argv[4]) : -0.0f;
  char* fileName = (argc > 5) ? argv[5] : "testOutput.ppm";
  const double scale = 4.0f / width;
  set = new MandelbrotSet(width, height, { real, imaginary });
  for (int i = 0; i < 5; i++) {
	  cout << "Attempt [" << i << "] " << endl;
	  measure(process, set, scale, fileName);
  }
  delete set;
  //std::cin.get();
  return 0;
}
