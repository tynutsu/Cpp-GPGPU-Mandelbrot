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
	float x = (column - width/2) * scale - 0.6f;
	float y = (row - height/2) * scale + 0.0f;
	float zx, zy, xx = x*x, yy = y*y;
	unsigned char iter = 0;

	zx = hypot(x - 0.25f, y);
	if (x < zx - 2 * zx * zx + .25 || (x + 1)*(x + 1) + yy < 1 / 16) {
		data[index].r = 0; data[index].g = 0; data[index].b = 0;
		return;
	}
	zx = 0.0f; zy = 0.0f;
	do
	{
		xx = zx * zx;
		yy = zy * zy;
		zy = 2 * zx * zy + y;
		zx = xx - yy + x;;
	} while ((xx + yy <= 4.0f) && (iter++ < MAXIT));
	if (iter == MAXIT || iter == 0) {
		data[index].r = 0; data[index].g = 0; data[index].b = 0;
	}
	else {
		data[index] = shades[iter % TOTAL_SHADES];
	}	
}

void process(MandelbrotSet* set, double scale, char* fileName, dim3 blocks, dim3 threads) {
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	int w = set->getWidth();
	int h = set->getHeight();
	Complex n = set->getComplex();
	cudaEventRecord(start);
	calc_mandel << <blocks, threads>> >(set->getDeviceReference(), w, h, scale, n);
	set->fetch();
	cudaEventSynchronize(end);
	float miliseconds = 0;
	cudaEventElapsedTime(&miliseconds, start, end);

	cout << "CUDA MEASURE TIME: " << miliseconds << endl;
}

void calculateKernelLimits(int& blockSize, int& minGridSize, int& gridSize) {

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
  long time = 0.0f;
  int threads = 32;
  dim3 blockSize(threads, threads);
  //dim3 gridSize(64,64);
  dim3 gridSize(width / threads, height / threads);
  for (int i = 0; i < 5; i++) {
	  cout << "Attempt [" << i << "] " << endl;
	  time+= measure(process, set, scale, fileName, gridSize, blockSize);
  }
  set->saveAs(fileName);
  cout << "Done in: " << time / 5 << " milliseconds";
  delete set;
  std::cin.get();
  return 0;
}
