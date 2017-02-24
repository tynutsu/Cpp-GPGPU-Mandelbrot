#include "Util.h"

void check(cudaError err, string message, string errorMessage, bool log) {
	if (err != 0) {
		std::cout << errorMessage << ": ";
		std::cout << cudaGetErrorString(err) << __FILE__ << __LINE__ << endl;
	}
	else {
		if (log) {
			std::cout << message << endl;
		}
	}
}