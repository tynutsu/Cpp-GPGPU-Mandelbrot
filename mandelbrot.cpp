#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <chrono>
#include <iostream>

using uchar = unsigned char;
using namespace std::chrono;
const double cx = -.6, cy = 0;
const uchar max_iter = std::numeric_limits<uchar>::max();

struct rgb_t { unsigned char r, g, b; };

rgb_t **row_ptrs;
rgb_t  *img_data;
 
void screen_dump(const int width, const int height)
{
  FILE *fp = fopen("out_mandelbrot.ppm", "w");
  fprintf(fp, "P6\n%d %d\n255\n", width, height);
  for (int i = height - 1; i >= 0; i--)
    fwrite(row_ptrs[i], 1, width * sizeof(rgb_t), fp);
  fclose(fp);
}
 
void alloc_2d(const int width, const int height)
{
  img_data = new rgb_t[width * height];
  row_ptrs = new rgb_t*[height];

  row_ptrs[0] = img_data;
  for (int i = 1; i < height; i++)
    row_ptrs[i] = row_ptrs[i-1] + width;
}

void map_colour(rgb_t * const px) 
{
  const uchar num_shades = 16;
  const rgb_t mapping[num_shades] =
    {{66,30,15},   {25,7,26},    {9,1,47},     {4,4,73},     {0,7,100},
     {12,44,138},  {24,82,177},  {57,125,209}, {134,181,229},{211,236,248},
     {241,233,191},{248,201,95}, {255,170,0},  {204,128,0},  {153,87,0},
     {106,52,3}};

  if (px->r == max_iter || px->r == 0) {
    px->r = 0; px->g = 0; px->b = 0;
  } else {
    const uchar uc = px->r % num_shades;
    *px = mapping[uc];
  }
}
 
void calc_mandel(const int width, const int height, const double scale)
{
  for (int i = 0; i < height; i++) {

    const double y = (i - height/2) * scale + cy;

    rgb_t *px = row_ptrs[i];
    for (int j = 0; j  < width; j++, px++) {

      const double x = (j - width/2) * scale + cx;
      double zx, zy, zx2, zy2;
      uchar iter = 0;
 
      zx = hypot(x - .25, y);
      if (x < zx - 2 * zx * zx + .25)     iter = max_iter;
      if ((x + 1)*(x + 1) + y * y < 1/16) iter = max_iter;
 
      zx = zy = zx2 = zy2 = 0;
      do {
        zy = 2 * zx * zy + y;
        zx = zx2 - zy2 + x;
        zx2 = zx * zx;
        zy2 = zy * zy;
      } while (iter++ < max_iter && zx2 + zy2 < 4);

      px->r = iter;
      px->g = iter;
      px->b = iter;
	  map_colour(px);
    }
  }
}

void run(int width, int height, double scale) {
	alloc_2d(width, height);
	calc_mandel(width, height, scale);
	screen_dump(width, height);
}
 
int main(int argc, char *argv[])
{
  const int width  = (argc > 1) ? std::atoi(argv[1]) : 4096;
  const int height = (argc > 2) ? std::atoi(argv[2]) : 4096;
  const double scale = 1./(width/4);
  for (int i = 0; i < 5; i++) {
	  high_resolution_clock::time_point start = high_resolution_clock::now();
	  run(width, height, scale);
	  high_resolution_clock::time_point end = high_resolution_clock::now();
	  auto elapsed = duration_cast<milliseconds> (end - start);
	  std::cout << "Execution time: " << elapsed.count() << " miliseconds " << i << std::endl;
  }
  std::cin.get();
  delete [] img_data;
  delete [] row_ptrs;
  return 0;
}
