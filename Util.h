#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <chrono>
#include <limits>
#include <fstream>
#include <istream>
#include <cmath>
#include <map>

using namespace std;
using namespace chrono;

#ifndef TOTAL_SHADES
#define TOTAL_SHADES 16
#endif

#ifndef LOG
#define LOG false
#endif

// upper limit for which time unit is displayed from milliseconds to nanoseconds
#ifndef NANOLIMIT
#define NANOLIMIT 15
#endif

// number of cycles to run one kernel to calculate average
#ifndef CYCLES
#define CYCLES 1
#endif

// just for better reading the next two
#ifndef ADD_RECOMMENDED_SUFFIX
#define ADD_RECOMMENDED_SUFFIX true
#endif

#ifndef SKIP_RECOMMENDED_SUFFIX
#define SKIP_RECOMMENDED_SUFFIX false
#endif

#ifndef SATURATE_KERNEL_CONFIGURATION
#define SATURATE_KERNEL_CONFIGURATION true
#endif


struct Pixel { unsigned char r, g, b; };
struct Complex { double real, imaginary; };
struct KernelProperties { dim3 gridSize, blockSize; };

const int POWERS[13] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 };

const map<int, dim3> SQRT = {	// TOTAAL THREADS; x,   y,   z
	{4096, dim3(64,64)},				// 4096 = 64 x 64 x  1
	{2048, dim3(32,32,2)},				// 2048 = 32 x 32 x  2
	{1024, dim3(32,32)},				// 1024 = 32 x 32 x  1
	{512, dim3(16,16,2)},				//  512 = 16 x 16 x  2
	{256, dim3(16,16)},					//  256 = 16 x 16 x  1
	{128, dim3(8,8,2)},					//  128 =  8 x  8 x  1
	{64, dim3(8,8)},					//   64 =  8 x  8 x  1
	{32, dim3(4,4,2)},					//   32 =  4 x  4 x  2
	{16, dim3(4,4)},					//   16 =  4 x  4 x  1
	{8, dim3(2,2,2)},					//    8 =  2 x  2 x  2
	{4, dim3(2,2)},						//    4 =  2 x  2 x  1
	{2, dim3{2,1}},						//    2 =  2 x  1 x  1
	{1, dim3(1)}						//    1 =  1 x  1 x  1
};

const unsigned char MAXIT = std::numeric_limits<unsigned char>::max();
const int BUFFER_SIZE = 4096;
struct Time { long long nano, millis; };


/*
	method used to personalise the cuda error message or to log successful execution
*/
void check(cudaError_t err, string message, string errorMessage = "", bool log = true);

/*
	template used to measure any method
	original code from http://stackoverflow.com/questions/22387586/measuring-execution-time-of-a-function-in-c
*/
template<typename M, typename... P>
Time measure(M method, P&&... parameters) {
	auto start = high_resolution_clock::now();
	method(forward<P>(parameters)...);
	auto end = high_resolution_clock::now();
	auto nanos = duration_cast<nanoseconds> (end - start).count();
	auto millis = duration_cast<milliseconds> (end - start).count();
	millis > NANOLIMIT ? cout << "Done in " << millis << " milliseconds" << endl : cout << "Done in " << nanos << " nanoseconds" << endl;
	return { nanos, millis };
}

// using occupancy api from:
// https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/

template<class T>
KernelProperties calculateKernelLimits(int width, int height, T function) {
	int blockSize;   // The launch configurator returned block size
	int minGridSize; // The minimum grid size needed to achieve the
					 // maximum occupancy for a full device launch
	int gridSize;    // The actual grid size needed, based on input size

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, function, 0, 0);
	// Round up according to array size

	gridSize = (width + blockSize - 1) / blockSize;
	printf("\nCalculating the best configuration\n");
	printf("Grid size: %d; Block Size: %d\n", gridSize, blockSize);
	// calculate theoretical occupancy
	int maxActiveBlocks;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, function, blockSize, 0);

	int device;
	cudaDeviceProp props;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&props, device);

	float occupancy = (float)(maxActiveBlocks * blockSize) / (props.maxThreadsPerMultiProcessor);
	printf("Launched grid of size %d, with %d threads. Theoretical occupancy: %f\n", gridSize, blockSize, occupancy);
	// spread the block size getting a precomputed sqrt or precomputed configuration for non square numbers
	if (SATURATE_KERNEL_CONFIGURATION) {
		dim3 blocks = SQRT.find(gridSize)->second;
		dim3 threads = SQRT.find(blockSize)->second;
		printf("Using saturation: grid layout [%i,%i,%i], threads layout [%i,%i,%i]\n", blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
		return{ blocks , threads };
	}
	else {
		return{ dim3(gridSize, gridSize), dim3(blockSize,blockSize) };
	}
};

