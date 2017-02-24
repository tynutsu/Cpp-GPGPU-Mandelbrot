#pragma once
#include "Util.h"

class CpuMandel {
	int width = 4096;
	int height = 4096;
	float scale = 4.0f / width;
	Pixel shades[TOTAL_SHADES] = {
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
	const double cx = -.6, cy = 0;
	Pixel **row_ptrs;
	Pixel  *img_data;
private:
	void screen_dump(const int width, const int height);
	void alloc_2d(const int width, const int height);
	void map_colour(Pixel * const px);
	void calc_mandel(const int width, const int height, const double scale);
	void run();
public:
	CpuMandel();
	CpuMandel(int w, int h, float s);
	~CpuMandel();	
	Pixel* getImage() { return img_data; }
};

