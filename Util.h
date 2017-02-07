#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <chrono>
#include <limits>

#ifndef TOTAL_SHADES
#define TOTAL_SHADES 16
#endif

struct Pixel { unsigned char r, g, b; };

using namespace std;
using namespace chrono;

const unsigned char maxit = std::numeric_limits<unsigned char>::max();


/*
	method used to personalise the cuda error message or to log successful execution
*/
void check(cudaError_t err, string message, string errorMessage = "");

/*
	template used to measure any method
	original code from http://stackoverflow.com/questions/22387586/measuring-execution-time-of-a-function-in-c
*/
template<typename M, typename... P>
void measure(M method, P&&... parameters) {
	auto start = high_resolution_clock::now();
	method(forward<P>(parameters)...);
	auto end = high_resolution_clock::now();
	auto elapsed = duration_cast<milliseconds> (end - start);
	cout << "Execution time: " << elapsed.count() << " miliseconds " << endl;
}