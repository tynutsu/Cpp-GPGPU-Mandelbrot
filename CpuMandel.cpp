#include "CpuMandel.h"

CpuMandel::CpuMandel() {
	run();
}

CpuMandel::CpuMandel(int w, int h, float s) {
	this->width = w;
	this->height = h;
	this->scale = s;
	run();
}


CpuMandel::~CpuMandel() {
	delete[] img_data;
	delete[] row_ptrs;
}

void CpuMandel::map_colour(Pixel * const px) {
	if (px->r == MAXIT || px->r == 0) {
		px->r = 0; px->g = 0; px->b = 0;
	}
	else {
		const unsigned char uc = px->r % TOTAL_SHADES;
		*px = shades[uc];
	}
}

void CpuMandel::screen_dump(const int width, const int height) {
	FILE *fp = fopen("cpuOutput.ppm", "w");
	fprintf(fp, "P6\n%d %d\n255\n", width, height);
	for (int i = height - 1; i >= 0; i--) {
		fwrite(row_ptrs[i], 1, width * sizeof(Pixel), fp);
	}
	fclose(fp);
}

void CpuMandel::alloc_2d(const int width, const int height) {
	img_data = new Pixel[width * height];
	row_ptrs = new Pixel*[height];

	row_ptrs[0] = img_data;
	for (int i = 1; i < height; i++) {
		row_ptrs[i] = row_ptrs[i - 1] + width;
	}
}

void CpuMandel::calc_mandel(const int width, const int height, const double scale) {
	for (int i = 0; i < height; i++) {

		const double y = (i - height / 2) * scale + cy;

		Pixel *px = row_ptrs[i];
		for (int j = 0; j < width; j++, px++) {

			const double x = (j - width / 2) * scale + cx;
			double zx, zy, zx2, zy2;
			unsigned char iter = 0;

			zx = hypot(x - .25, y);
			if (x < zx - 2 * zx * zx + .25)     iter = MAXIT;
			if ((x + 1)*(x + 1) + y * y < 1 / 16) iter = MAXIT;

			zx = zy = zx2 = zy2 = 0;
			do {
				zy = 2 * zx * zy + y;
				zx = zx2 - zy2 + x;
				zx2 = zx * zx;
				zy2 = zy * zy;
			} while (iter++ < MAXIT && zx2 + zy2 < 4);
			if (iter == MAXIT || iter == 0) {
				px->r = 0; px->g = 0; px->b = 0;
			}
			else {
				*px = shades[iter % TOTAL_SHADES];
			}
		}
	}
}

void CpuMandel::run() {
	alloc_2d(width, height);
	calc_mandel(width, height, scale);
	screen_dump(width, height);
}

