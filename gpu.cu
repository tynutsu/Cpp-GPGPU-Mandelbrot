#include <cmath>
#include "MandelbrotSet.h"
#include <fstream>

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
	int w = set->getWidth();
	int h = set->getHeight();
	Complex n = set->getComplex();
	calc_mandel << <blocks, threads>> >(set->getDeviceReference(), w, h, scale, n);
	set->fetch();
}

// using occupancy api from:
// https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
KernelProperties calculateKernelLimits(int width, int height) {
	int blockSize;   // The launch configurator returned block size
	int minGridSize; // The minimum grid size needed to achieve the
					 // maximum occupancy for a full device launch
	int gridSize;    // The actual grid size needed, based on input size

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calc_mandel, 0, 0);
	// Round up according to array size

	gridSize = (width + blockSize - 1) / blockSize;
	printf("Grid size: %d; Block Size: %d\n", gridSize, blockSize);
	// calculate theoretical occupancy
	int maxActiveBlocks;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, calc_mandel, blockSize, 0);

	int device;
	cudaDeviceProp props;
	printf("Max grid size: %d\n", props.maxGridSize);
	printf("Max block size: %d\n", props.maxThreadsPerBlock);
	printf("Max shared memory per block: %d\n", props.sharedMemPerBlock);
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&props, device);

	float occupancy = (maxActiveBlocks * blockSize / props.warpSize) / (float)(props.maxThreadsPerMultiProcessor / props.warpSize);

	printf("Launched grid of size %d, with %d threads. Theoretical occupancy: %f\n", gridSize, blockSize, occupancy);
	return{ gridSize, blockSize };
}

const int POWERS[13] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 };

int main(int argc, char *argv[])
{
	
  const int width  = (argc > 1) ? std::atoi(argv[1]) : 4096;
  const int height = (argc > 2) ? std::atoi(argv[2]) : 4096;
  double real = (argc > 3) ? std::atol(argv[3]) : -0.6f;
  double imaginary = (argc > 4) ? std::atol(argv[4]) : -0.0f;
  char* fileName = (argc > 5) ? argv[5] : "testOutput.ppm";
  const double scale = 4.0f / width;
  string logName = to_string(width)+"x"+to_string(height)+"-log.csv";
  fstream data(logName);
  data << "Width, Height, Blocks, Threads, Time, Millis, Nanos";
  set = new MandelbrotSet(width, height, { real, imaginary });
  long time = 0.0f;
  string unit = "";

  for (int power = 0; power < 13; power++) {
	  int threads = POWERS[power];
	  string name = "";
	  dim3 blockSize(threads, threads);
	  dim3 gridSize(width / threads, height / threads);
	  time = 0.0f;
	  for (int i = 0; i < CYCLES; i++) {
		  data << width << "," << height << "," << width / threads << "," << threads << ",";
		  cout << "Attempt [" << i << "] " << gridSize.x << " blocks; " << blockSize.x << " threads: ";
		  auto value = measure(process, set, scale, fileName, gridSize, blockSize);
		  if (value.millis > NANOLIMIT) {
			  time += value.millis; 
			  unit = "_milliseconds_";
			  
		  }
		  else {
			  time += value.nano;
			  unit = "_nanoseconds_";
		  }
		  data << value.millis << value.nano << "\n";
	  }
	  cout << "Average time: " << time / CYCLES << endl;
	  name.append(to_string(width) + "x" + to_string(height) + "_Blocks_" + to_string(width / threads) + "_Threads_" + to_string(threads) + unit + to_string(time / CYCLES) + ".ppm");
	  cout << endl << "FILENAME IS: " << name << endl;
	  set->saveAs(name);
  }
  KernelProperties suggested = calculateKernelLimits(width, height);
  dim3 gridSize(suggested.gridSize, suggested.gridSize);
  dim3 blockSize(suggested.blockSize, suggested.blockSize);
  time = 0;
  for (int i = 0; i < CYCLES; i++) {
	  data << width << "," << height << "," << suggested.gridSize << "," << suggested.blockSize << ",";
	  cout << "Attempt [" << i << "] " << gridSize.x << " blocks; " << blockSize.x << " threads: ";
	  auto value = measure(process, set, scale, fileName, gridSize, blockSize);
	  if (value.millis > NANOLIMIT) {
		  time += value.millis;
		  unit = "_milliseconds_";
	  }
	  else {
		  time += value.nano;
		  unit = "_nanoseconds_";
	  }
	  data << value.millis << value.nano << "RECOMMENDED\n";
  }
  string name = "";
  name.append(to_string(width) + "x" + to_string(height) + "_Blocks_" + to_string(suggested.gridSize) + "_Threads_" + to_string(suggested.blockSize) + unit + to_string(time / CYCLES) + "___RECOMMENDED.ppm");
  cout << "\nRecommended settings: " << suggested.gridSize << " blocks, " << suggested.blockSize << " threads completed in " << time / CYCLES << " " << unit << endl;
  set->saveAs(name);
  data.close();

  delete set;
  std::cin.get();
  return 0;
}