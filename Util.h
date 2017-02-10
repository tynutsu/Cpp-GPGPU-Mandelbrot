#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <chrono>
#include <limits>

#ifndef TOTAL_SHADES
#define TOTAL_SHADES 16
#endif

#ifndef NANOLIMIT
#define NANOLIMIT 13
#endif

#ifndef CYCLES
#define CYCLES 5
#endif

struct Pixel { unsigned char r, g, b; };
struct Complex { double real, imaginary; };
struct KernelProperties { int gridSize, blockSize; };

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
	return{ nanos, millis };
}