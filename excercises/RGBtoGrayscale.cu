#include <cuda_runtime.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define CHANNELS 3

__global__ void colorToGray(unsigned char* Pout, unsigned char* Pin, int width, int height)
{
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;

    if (Col < width && Row < height) {
        int greyOffset = Row * width + Col;
        int rgbOffset = greyOffset * CHANNELS;
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];

        Pout[greyOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast (const char* const file, const int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error ar:"<< file<<line<<std::endl;
        std::cerr << cudaGetErrorString(err)<< std::endl;
        std::exit(EXIT_FAILURE);
    }
}


//lettura file png:

int read_png(const char* filename, unsigned char** image, int* width, int* height) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        return -1;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(fp);
        return -1;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, NULL, NULL);
        fclose(fp);
        return -1;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        return -1;
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    *width = png_get_image_width(png, info);
    *height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if (bit_depth == 16) {
        png_set_strip_16(png);
    }

    if (color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png);
    }

    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
        png_set_expand_gray_1_2_4_to_8(png);
    }

    if (png_get_valid(png, info, PNG_INFO_tRNS)) {
        png_set_tRNS_to_alpha(png);
    }

    if (color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
        png_set_gray_to_rgb(png);
    }

    png_read_update_info(png, info);

    size_t rowbytes = png_get_rowbytes(png, info);
    *image = (unsigned char*)malloc((*height) * rowbytes);
    png_bytep row_pointers[*height];
    for (int y = 0; y < *height; y++) {
        row_pointers[y] = *image + y * rowbytes;
    }

    png_read_image(png, row_pointers);

    fclose(fp);
    png_destroy_read_struct(&png, &info, NULL);

    return 0;
}


//scrittura file png:

int write_png(const char* filename, unsigned char* image, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        return -1;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(fp);
        return -1;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, NULL);
        fclose(fp);
        return -1;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return -1;
    }

    png_init_io(png, fp);

    png_set_IHDR(
        png,
        info,
        width, height,
        8,
        PNG_COLOR_TYPE_GRAY,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png, info);

    png_bytep row_pointers[height];
    for (int y = 0; y < height; y++) {
        row_pointers[y] = image + y * width;
    }

    png_write_image(png, row_pointers);
    png_write_end(png, NULL);

    fclose(fp);
    png_destroy_write_struct(&png, &info);

    return 0;
}


int main(int argc, char** argv) 
{
    const char* input_filename = argv[1];
    const char* output_filename = argv[2];

    unsigned char* image;
    int width, height;

    read_png(input_filename, &image, &width, &height);

    size_t rgb_size = width * height * CHANNELS * sizeof(unsigned char);
    size_t grey_size = width * height * sizeof(unsigned char);

    unsigned char *d_rgb, *d_grey;
  
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    CHECK_LAST_CUDA_ERROR();
    colorToGray<<<gridSize, blockSize>>>(d_grey, d_rgb, width, height);
    
    CHECK_LAST_CUDA_ERROR();

    unsigned char* output_image = (unsigned char*)malloc(grey_size);
    
    CHECK_LAST_CUDA_ERROR();

    if (write_png(output_filename, output_image, width, height) != 0) {
        printf("Error writing PNG file\n");
        return 1;
    }

    cudaFree(d_rgb);
    cudaFree(d_grey);

    CHECK_LAST_CUDA_ERROR();

    free(image);
    free(output_image);

    printf("Image successfully converted and saved as %s\n", output_filename);

    return 0;
}