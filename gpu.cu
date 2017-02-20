#include "MandelbrotSet.h"
#include "CpuMandel.h"

// make the application aware that there will be some definitions later on
bool isEqual(Pixel* first, Pixel* second);
void compare(Pixel* first, Pixel* second, int width, int height);
void process(MandelbrotSet* set, double scale, dim3 blocks, dim3 threads);
string logStats(int times, dim3 blocks, dim3 threads, MandelbrotSet* set, float scale, fstream &statsFile, bool recommended);
/*
function to create filename that tracks the image size, kernel configuration and execution time
w - width
h - height
t - threads
a - average
unit - unit (milliseconds or nanoseconds)
*/
string nameFile(int width, int height, dim3 threads, long time, string unit, bool recommended);

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

__constant__ Pixel black = { 0, 0, 0 };

// MandelBrot algorithm on __device__
__global__ void calc_mandel(Pixel  *data, const int width, const int height, const double scale, const Complex number)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int index = row * width + column;
	const float x = (column - width/2) * scale - 0.6f;
	const float y = (row - height/2) * scale + 0.0f;
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
	data[index] = (iter == MAXIT || iter == 0) ? black : shades[iter % TOTAL_SHADES];
}

int main(int argc, char *argv[])
{
	/* parse arguments 
	argv[1] = width
	argv[2] = height
	argv[3] = compare cpu vs gpu outputs? "false" or no argument for false or anything else for true;
	argv[4] = real part of complex number
	argv[5] = imaginary part of complex number
	*/
	const auto width  = argc > 1 ? std::atoi(argv[1]) : 4096;
	const auto height = argc > 2 ? std::atoi(argv[2]) : 4096;
	auto shouldCompare = argc > 3 ? true : false;
	if (shouldCompare) {
		string value(argv[3]);
		shouldCompare = value == "false" ? false : true;
	};
	auto real = argc > 4 ? std::atol(argv[4]) : -0.6f;
	auto imaginary = argc > 5 ? std::atol(argv[5]) : -0.0f;
	const auto scale = 4.0f / width;

	// open valid csv file for writing and add headers or exit
	fstream outputStatsFile(to_string(width) + "x" + to_string(height) + "-log.csv", fstream::out);
	if (outputStatsFile.is_open()) {
		std::cout << std::endl << "SUCCESS OPENING FILE" << endl;
		outputStatsFile << "Width, Height, Blocks, Threads, Millis, Nanos\n";
	} else {
		std::cout << "CANNOT OPEN FILE " << to_string(width) + "x" + to_string(height) + "-log.csv";
		std::cout << std::endl << "Please make sure the file is not opened already in Excel.";
		return -1;
	}
	
	// create gpu mandelbrot set
	auto set = new MandelbrotSet(width, height, { real, imaginary });
	
	// generate sets output and log configurations for threads = power of 2 and blocks is depending on image width
	for (auto power = 0; power < 13; power++) {
		auto threads = POWERS[power];
		auto blocks = width / threads;
		set->saveAs(
			logStats(
				CYCLES, 
				dim3(blocks, blocks), 
				dim3(threads,threads),
				set,
				scale, 
				outputStatsFile, 
				SKIP_RECOMMENDED_SUFFIX));
	}
	
	// generate set output for recommended settings using the CUDA occupancy API
	KernelProperties recommended = calculateKernelLimits(width, height, calc_mandel);
	set->saveAs(
		logStats(
			CYCLES, 
			recommended.gridSize, 
			recommended.blockSize, 
			set,
			scale, 
			outputStatsFile, 
			ADD_RECOMMENDED_SUFFIX));
	
	if (shouldCompare) {
		cout << "Comparing the two outputs" << endl;
		auto cpuSet = new CpuMandel(width, height, scale);
		compare(cpuSet->getImage(), set->getHostReference(), width, height);
		delete cpuSet;
	} else {
		cout << endl << "Skipping output comparison by default; add 3rd argument to compare outputs i.e.: gpu.exe 4096 4096 true" << endl;
	}

	outputStatsFile.close();
	delete set;
	std::cin.get();
	return 0;
}

bool isEqual(Pixel* first, Pixel* second) {
	return first->b == second->b
		&& first->g == second->g
		&& first->r == second->r;
};

void compare(Pixel* first, Pixel* second, int width, int height) {
	int size = width * height;
	for (int i = 0; i < size; i++) {
		if (!isEqual(first, second)) {
			cout << endl << "Images are different";
			cin.get();
			return;
		}
		if (i && i % PIXEL_COUNT_REPORT == 0) {
			cout << endl << "So far " << i << " pixels are identical";
		}
	}
	cout << endl << "All " << size << " pixels are identical";
	cout << endl << "Images are identical";
	cin.get();
};

void process(MandelbrotSet* set, double scale, dim3 blocks, dim3 threads) {
	calc_mandel << <blocks, threads >> >	(
		set->getDeviceReference(),
		set->getWidth(),
		set->getHeight(),
		scale,
		set->getComplex());
	set->fetch();
}

string nameFile(int width, int height, dim3 threads, long time, string unit, bool recommended) {
	const string ending = recommended ? "___RECOMMENDED.ppm" : ".ppm";
	string outputImageName = to_string(width) + "x"
		+ to_string(height) + "_Blocks_"
		+ to_string(width / threads.x) + "_Threads_"
		+ to_string(threads.x) + unit
		+ to_string(time / CYCLES)
		+ ending;
	std::cout << endl << "FILENAME IS: " << outputImageName << endl;
	return outputImageName;
}

string logStats(int times, dim3 blocks, dim3 threads, MandelbrotSet* set, float scale, fstream &statsFile, bool recommended) {
	long cycleTime = 0.0f;
	string unit = "";
	const string endline = recommended ? ", ,RECOMMENDED\n" : "\n";
	const int w = set->getWidth();
	const int h = set->getHeight();

	for (int i = 0; i < times; i++) {

		std::cout << "Attempt [" << i << "] " << blocks.x << " blocks; " << threads.x << " threads: ";
		auto value = measure(process, set, scale, blocks, threads);
		if (value.millis > NANOLIMIT) {
			cycleTime += value.millis;
			unit = "_milliseconds_";

		}
		else {
			cycleTime += value.nano;
			unit = "_nanoseconds_";
		}
		statsFile << w << "," << h << "," << w / threads.x << "," << threads.x << ",";
		statsFile << value.millis << "," << value.nano << endline;
	}
	std::cout << "Average time: " << cycleTime / times << endl;
	return nameFile(w, h, threads.x, cycleTime, unit, recommended);
}