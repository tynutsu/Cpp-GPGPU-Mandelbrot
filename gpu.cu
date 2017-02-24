#include "MandelbrotSet.h"
#include "CpuMandel.h"

// make the application aware that there will be some definitions later on
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
bool compareFiles(const string& firstImageName, const string& secondImageName);

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

// MandelBrot algorithm on __device__
__global__ void calc_mandel(Pixel  *data, const int width, const int height, const double scale, const Complex number) {
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int index = (height - row - 1) * width + column; // reverse order

	if (column >= width || row >= height) {
		return;
	}

	double x0 = (column - width / 2) * scale + number.real;
	double y0 = (row - height / 2) * scale + number.imaginary;
	double x = hypot(x0 - 0.25, y0);
	double y = y0 * y0;		// using it here to initialize directly with the squre of y, will be used as normal later
	double xx = x * x;
	double yy = (x0 + 1)*(x0 + 1);	// temporary use as well

	if (x - 2 * xx + 0.25 >= x0 || yy + y < 0.0625) {
		return;
	}
	
	x = y = xx = yy = 0.0; // resetting and using as normal
	uint8_t iter = 0;

	do {
		y = 2 * x * y + y0;
		x = xx - yy + x0;
		xx = x * x;
		yy = y * y;
	} while (iter++ < MAXIT && xx + yy < 4.0);

	if (iter != MAXIT && iter != 0) {
		data[index] = shades[iter % TOTAL_SHADES];
	}
	
}

int main(int argc, char *argv[]) {
	/* parse arguments 
	argv[1] = width
	argv[2] = height
	argv[3] = compare cpu vs gpu outputs? "false" or no argument for false or anything else for true;
	argv[4] = real part of complex number
	argv[5] = imaginary part of complex number
	*/
	const auto width  = argc > 1 ? std::atoi(argv[1]) : 4096;
	const auto height = argc > 2 ? std::atoi(argv[2]) : 4096;
	bool shouldCompare = argc > 3 ? true : false; 
	if (shouldCompare) {
		string value(argv[3]);
		shouldCompare = value == "true" ? true : false;
	}
	const double real = argc > 4 ? std::atol(argv[4]) : -0.6;
	const double imaginary = argc > 5 ? std::atol(argv[5]) : -0.0;
	const float scale = 4.0f / width;

	// open valid csv file for writing and add headers or exit
	fstream outputStatsFile(to_string(width) + "x" + to_string(height) + "-log.csv", fstream::out);
	if (outputStatsFile.is_open()) {
		std::cout << std::endl << "SUCCESS OPENING FILE" << endl;
		outputStatsFile << "Width, Height\n";
		outputStatsFile << width << "," << height << "\n";
		outputStatsFile << "Blocks, Grid Layout, Threads, Block Layout, Millis, Nanos\n";
	} else {
		std::cout << "CANNOT OPEN FILE " << to_string(width) + "x" + to_string(height) + "-log.csv";
		std::cout << std::endl << "Please make sure the file is not opened already in Excel.";
		return -1;
	}
	
	// create gpu mandelbrot set
	MandelbrotSet* setGPU = new MandelbrotSet(width, height, { real, imaginary });
	
	// generate sets output and log configurations for threads = power of 2 and blocks is depending on image width
	for (auto power = 0; power < 13; power++) {
		int threads = POWERS[power];
		if (threads > width) {
			break;
		}
		int blocks = width / threads;
		setGPU->saveAs(
			logStats(
				CYCLES, 
				dim3(blocks, blocks),
				dim3(threads, threads),
				setGPU,
				scale, 
				outputStatsFile, 
				SKIP_RECOMMENDED_SUFFIX));
	}
	
	// generate set output for recommended settings using the CUDA occupancy API
	KernelProperties recommended = calculateKernelLimits(width, height, calc_mandel);
	setGPU->saveAs(
		logStats(
			CYCLES, 
			recommended.gridSize, 
			recommended.blockSize, 
			setGPU,
			scale, 
			outputStatsFile, 
			ADD_RECOMMENDED_SUFFIX));

	outputStatsFile.close();
	if (shouldCompare) {
		CpuMandel* setCPU = new CpuMandel(width, height, scale);
		compareFiles("cpuOutput.ppm", "recommended.ppm");
		delete setCPU;
	}
	delete setGPU;
	
	std::cin.get();
	return 0;
}

// simple wrapper to send to the measure template which performs a fetch at the end
void process(MandelbrotSet* set, double scale, dim3 blocks, dim3 threads) {
	calc_mandel << <blocks, threads >> >	(
		set->getDeviceReference(),
		set->getWidth(),
		set->getHeight(),
		scale,
		set->getComplex());
	set->fetch();
}

// function to build the name of the image containing kernel configuration and other details
string nameFile(int width, int height, dim3 threads, long long time, string unit, bool recommended) {
	// if recommended save as "recommended.ppm" else build the imageName
	string outputImageName = recommended ? "recommended.ppm" : to_string(width) + "x"
		+ to_string(height) + "_Blocks_"
		+ to_string(width / threads.x) + "_Threads_"
		+ to_string(threads.x) + unit
		+ to_string(time / CYCLES)
		+ ".ppm";
	std::cout << endl << "FILENAME IS: " << outputImageName << endl;
	return outputImageName;
}

// function to log time into a excel file format (comma separated values)
string logStats(int times, dim3 blocks, dim3 threads, MandelbrotSet* set, float scale, fstream &statsFile, bool recommended) {
	long long cycleTime = 0;
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
		int gridSize = blocks.x * blocks.y * blocks.z;
		int blockSize = threads.x * threads.y * threads.z;
		statsFile << gridSize << ",[" << blocks.x << "-" << blocks.y << "-" << blocks.z << "],";
		statsFile << blockSize << ",[" << threads.x << "-" << threads.y << "-" << threads.z << "],";
		statsFile << value.millis << "," << value.nano << endline;
	}
	std::cout << "Average time: " << cycleTime / times << endl;
	return nameFile(w, h, threads.x, cycleTime, unit, recommended);
};

// using Quick comparison of two files code from http://www.cplusplus.com/forum/general/94032/
bool compareFiles(const string& firstImageName, const string& secondImageName)
{
	ifstream firstFile(firstImageName.c_str(), ifstream::in | ifstream::binary);
	ifstream secondFile(secondImageName.c_str(), ifstream::in | ifstream::binary);

	if (!firstFile.is_open() || !secondFile.is_open())
	{
		cout << "\n\n\nOne of the files could not be opened... aborting comparison\n\n\n";
		return false;
	}

	char *firstBuffer = new char[BUFFER_SIZE]();
	char *secondBuffer = new char[BUFFER_SIZE]();

	do {
		firstFile.read(firstBuffer, BUFFER_SIZE);
		secondFile.read(secondBuffer, BUFFER_SIZE);

		if (std::memcmp(firstBuffer, secondBuffer, BUFFER_SIZE) != 0)
		{
			delete[] firstBuffer;
			delete[] secondBuffer;
			cout << "\n\n\nFiles are different!\n\n\n";
			return false;
		}
	} while (firstFile.good() || secondFile.good());

	delete[] firstBuffer;
	delete[] secondBuffer;
	cout << "\n\All good, files are identical\n";
	return true;
};