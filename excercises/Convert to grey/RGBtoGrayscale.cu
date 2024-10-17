#include <cuda_runtime.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define CHANNELS 3

__global__ void colorToGray(unsigned char* Pout, unsigned char* Pin, int width, int height, int rowbytes)
{
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;

    if (Col < width && Row < height) 
    {
        // Usare rowbytes per il calcolo dell'offset
        int greyOffset = Row * width + Col;
        int rgbOffset = Row * rowbytes + Col * CHANNELS;

        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];

        // Convert RGB to grayscale using the luminance formula
        Pout[greyOffset] = static_cast<unsigned char>(0.21f * r + 0.71f * g + 0.07f * b);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at:" << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Function to read PNG file
int read_png(const char* filename, unsigned char** image, int* width, int* height, int* rowbytes_out) {
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

    // Handle special formats
    if (bit_depth == 16) {
        png_set_strip_16(png); // Strip 16-bit channels down to 8 bits
    }

    if (color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png); // Convert palette to RGB
    }

    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
        png_set_expand_gray_1_2_4_to_8(png); // Expand gray to 8-bit
    }

    if (png_get_valid(png, info, PNG_INFO_tRNS)) {
        png_set_tRNS_to_alpha(png); // Handle transparency
    }

    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
        png_set_gray_to_rgb(png); // Convert gray to RGB
    }

    png_read_update_info(png, info);

    // Get the number of bytes per row
    size_t rowbytes = png_get_rowbytes(png, info);
    *rowbytes_out = rowbytes;  // Return rowbytes to the caller

    // Allocate memory for the image
    *image = (unsigned char*)malloc((*height) * rowbytes);
    if (!*image) {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        return -1;
    }

    // Create row pointers
    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * (*height)); // Allocate memory for row pointers
    for (int y = 0; y < *height; y++) {
        row_pointers[y] = *image + y * rowbytes;
    }

    // Read the image
    png_read_image(png, row_pointers);

    // Free the row pointers
    free(row_pointers);

    // Close the file and free PNG structures
    fclose(fp);
    png_destroy_read_struct(&png, &info, NULL);

    return 0; // Success
}


// Function to write PNG file
int write_png(const char* filename, unsigned char* image, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("Error opening file for writing");
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

    // Create row pointers for grayscale image
    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    if (!row_pointers) {
        perror("Error allocating memory for row pointers");
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return -1;
    }

    for (int y = 0; y < height; y++) {
        row_pointers[y] = image + y * width;  // Use width for the correct offset
    }

    png_write_image(png, row_pointers);
    png_write_end(png, NULL);

    free(row_pointers); // Free memory allocated for row pointers
    fclose(fp);
    png_destroy_write_struct(&png, &info);

    return 0;
}


int main(int argc, char** argv) 
{
    if (argc < 3) {
        printf("Usage: %s <input_png> <output_png>\n", argv[0]);
        return 1;
    }

    const char* input_filename = argv[1];
    const char* output_filename = argv[2];

    unsigned char* image;
    int width, height, rowbytes;

    // Read the PNG image
    if (read_png(input_filename, &image, &width, &height, &rowbytes) != 0) {
        printf("Error reading PNG file\n");
        return 1;
    }

    size_t rgb_size = height * rowbytes;
    size_t grey_size = width * height * sizeof(unsigned char);

    unsigned char *d_rgb, *d_grey;

    // Allocate GPU memory
    if (cudaMalloc((void**)&d_rgb, rgb_size) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for RGB\n");
        free(image);
        return 1;
    }
    
    if (cudaMalloc((void**)&d_grey, grey_size) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for grayscale\n");
        cudaFree(d_rgb);  // Free previously allocated memory
        free(image);
        return 1;
    }

    // Copy the RGB image to GPU memory
    cudaMemcpy(d_rgb, image, rgb_size, cudaMemcpyHostToDevice);
    CHECK_LAST_CUDA_ERROR();
  
    // Set block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Execute the kernel
    colorToGray<<<gridSize, blockSize>>>(d_grey, d_rgb, width, height, rowbytes);
    CHECK_LAST_CUDA_ERROR();

    unsigned char* output_image = (unsigned char*)malloc(grey_size);

    if (output_image == NULL) {
        printf("Error allocating memory for output image\n");
        cudaFree(d_rgb);
        cudaFree(d_grey);
        free(image);
        return 1;
    }
    
    cudaMemcpy(output_image, d_grey, grey_size, cudaMemcpyDeviceToHost);
    CHECK_LAST_CUDA_ERROR();

    // Write the grayscale image to a PNG file
    if (write_png(output_filename, output_image, width, height) != 0) {
        printf("Error writing PNG file\n");
        free(output_image);
        cudaFree(d_rgb);
        cudaFree(d_grey);
        free(image);
        return 1;
    }

    // Free GPU memory
    cudaFree(d_rgb);
    cudaFree(d_grey);

    // Free host memory
    free(image);
    free(output_image);

    printf("Image successfully converted and saved as %s\n", output_filename);

    return 0;
}
