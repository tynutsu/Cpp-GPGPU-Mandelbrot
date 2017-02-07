#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <string>

using namespace std;

void check(cudaError_t err, string message, string errorMessage = "");