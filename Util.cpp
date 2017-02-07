#include "Util.h"

void check(cudaError err, string message, string errorMessage) {
	if (err != 0) {
		std::cout << errorMessage << ": ";
		std::cout << cudaGetErrorString(err) << __FILE__ << __LINE__ << endl;
	}
	else {
		std::cout << message << endl;
	}
}