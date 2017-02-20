#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <chrono>
#include <limits>
#include <fstream>
#include <cmath>

#ifndef TOTAL_SHADES
#define TOTAL_SHADES 16
#endif

#ifndef NANOLIMIT
#define NANOLIMIT 13
#endif

#ifndef CYCLES
#define CYCLES 5
#endif

#ifndef ADD_RECOMMENDED_SUFFIX
#define ADD_RECOMMENDED_SUFFIX true
#endif

#ifndef SKIP_RECOMMENDED_SUFFIX
#define SKIP_RECOMMENDED_SUFFIX false
#endif

#ifndef PIXEL_COUNT_REPORT
#define PIXEL_COUNT_REPORT 1000000
#endif


struct Pixel { unsigned char r, g, b; };
struct Complex { double real, imaginary; };
struct KernelProperties { dim3 gridSize, blockSize; };

const int POWERS[13] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 };

using namespace std;
using namespace chrono;

const unsigned char MAXIT = std::numeric_limits<unsigned char>::max();
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
	return{ dim3(gridSize, gridSize), dim3(blockSize,blockSize) };
};

