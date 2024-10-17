#include <cuda_runtime.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define CHANNELS 3
#define BLUR_SIZE 3



// Kernel fornito dal professore nelle slides, modificato per gestire il canale alpha
__global__ void blurKernel(unsigned char* out, const unsigned char* in, int width, int height, int pitch)
{
    //calcolo dello stride per accedere agli elementi dell'immagine (ai pixel)
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y; 

    if (Col < width && Row < height) // per non accedere a pixel fuori dall'immagine e andare in out of bounds
    {
        float pixR = 0, pixG = 0, pixB = 0, pixA = 0; // Aggiunto pixA per il canale alpha
        float pixels = 0; //numero di pixel considerati per la media: si considerano i vicinati di Moore per ogni pixel

        for (int blurRow = - BLUR_SIZE; blurRow <= BLUR_SIZE; blurRow++) // Modificato < a <= per includere l'ultimo pixel
        {
            for (int blurCol = - BLUR_SIZE; blurCol <= BLUR_SIZE; blurCol++) 
            {
                //coordinate dei pixel vicini (quello considerato al momento)
                int curRow = Row + blurRow; 
                int curCol = Col + blurCol;

                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) //controllare che il pixel sia all'interno dell'immagine
                {
                    int offset = curRow * pitch + curCol * CHANNELS; //posizione del pixel nell'array lineare che contiene i dati dell'immagine
                    //pitch è il numero di byte per riga, CHANNELS è il numero di canali (RGBA)
                    
                    float alpha = in[offset + 3] / 255.0f; // Normalizzazione del canale alpha
                    //accumulazione di rosso, verde, blu e alpha dei pixel vicini, pesati per il canale alpha
                    pixR += in[offset] * alpha; 
                    pixG += in[offset + 1] * alpha;
                    pixB += in[offset + 2] * alpha;
                    pixA += alpha;

                    pixels++; //incremento del numero di pixel considerati
                }
            }
        }

        int outOffset = Row * pitch + Col * CHANNELS; //posizione del pixel nell'array lineare che contiene i dati dell'immagine generata 
        
        //calcolo della media dei valori dei pixel vicini, gestendo correttamente il canale alpha
        if (pixA > 0) {
            out[outOffset] = (unsigned char)(pixR / pixA);
            out[outOffset + 1] = (unsigned char)(pixG / pixA);
            out[outOffset + 2] = (unsigned char)(pixB / pixA);
            out[outOffset + 3] = (unsigned char)(pixA / pixels * 255); // Normalizzazione inversa del canale alpha
        } else {
            out[outOffset] = 0;
            out[outOffset + 1] = 0;
            out[outOffset + 2] = 0;
            out[outOffset + 3] = 0;
        }
    }
}


// Funzione per la gestione degli errori CUDA
#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t const err{cudaGetLastError()}; //ottengo l'ultimo errore CUDA
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at:" << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Funzione per la lettura di un file PNG
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
    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * (*height));
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
        PNG_COLOR_TYPE_RGBA, // Modificato da GRAY a RGBA
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png, info);

    // Create row pointers for RGBA image
    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    if (!row_pointers) {
        perror("Error allocating memory for row pointers");
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return -1;
    }

    int rowbytes = width * CHANNELS; // Calcolo corretto dei bytes per riga
    for (int y = 0; y < height; y++) {
        row_pointers[y] = image + y * rowbytes;
    }

    png_write_image(png, row_pointers);
    png_write_end(png, NULL);

    free(row_pointers);
    fclose(fp);
    png_destroy_write_struct(&png, &info);

    return 0;
}

int main(int argc, char** argv) 
{
    if (argc < 3) 
    {
        printf("Usage: %s <input_png> <output_png>\n", argv[0]);
        return 1;
    } 
    //se non ci sono abbastanza argomenti, stampa il messaggio e ritorna 1

    const char* input_filename = argv[1]; //nome del file di input
    const char* output_filename = argv[2]; //nome del file di output

    unsigned char* image; //array che conterrà i dati dell'immagine
    int width, height, rowbytes; //larghezza, altezza e numero di byte per riga dell'immagine

    // Leggi l'immagine PNG
    if (read_png(input_filename, &image, &width, &height, &rowbytes) != 0) {
        printf("Error reading PNG file\n");
        return 1;
    }

    size_t image_size = height * rowbytes; //calcolo della dimensione dell'immagine

    unsigned char *d_input, *d_output; 

   // Alloca memoria GPU
    if (cudaMalloc((void**)&d_input, image_size) != cudaSuccess ||
        cudaMalloc((void**)&d_output, image_size) != cudaSuccess) 
    {
        fprintf(stderr, "Error allocating device memory\n");
        free(image);
        return 1;
    }

    // Copia l'immagine nella memoria GPU, dall'host al device
    cudaMemcpy(d_input, image, image_size, cudaMemcpyHostToDevice);
    CHECK_LAST_CUDA_ERROR();
  
    // Imposta le dimensioni del blocco e della griglia
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Esegui il kernel
    blurKernel<<<gridSize, blockSize>>>(d_output, d_input, width, height, rowbytes);
    CHECK_LAST_CUDA_ERROR();

    // Copia il risultato dalla GPU alla CPU
    cudaMemcpy(image, d_output, image_size, cudaMemcpyDeviceToHost);
    CHECK_LAST_CUDA_ERROR();

    // Scrivi l'immagine sfocata su un nuovo file PNG
    if (write_png(output_filename, image, width, height) != 0) {
        printf("Error writing PNG file\n");
        cudaFree(d_input);
        cudaFree(d_output);
        free(image);
        return 1;
    }

    // Libera la memoria GPU
    cudaFree(d_input);
    cudaFree(d_output);

    // Libera la memoria host
    free(image);

    printf("Image successfully blurred and saved as %s\n", output_filename);

    return 0;
}