#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define ROWS_A (1 << 11)  // 2048
#define COLS_A (1 << 10)  // 1024
#define ROWS_B COLS_A     // 1024
#define COLS_B (1 << 9)   // 512

// Kernel per l'inizializzazione della matrice
__global__ void matrixInit(float* matrix, int rows, int cols, float value) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < cols && idy < rows)
    {
        matrix[idy * cols + idx] = value;
    }
}

// Kernel per la moltiplicazione delle matrici (versione monolitica)
__global__ void matrixMultMonolithic(float* A, float* B, float* C, int rowsA, int colsA, int colsB) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rowsA && col < colsB)
    {
        float sum = 0.0f;
        for (int k = 0; k < colsA; ++k) 
        {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

// Kernel per la moltiplicazione delle matrici (versione grid-stride loop)
__global__ void matrixMultGridStride(float* A, float* B, float* C, int rowsA, int colsA, int colsB) 
{
    for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < rowsA; row += gridDim.y * blockDim.y)
    {
        for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < colsB; col += gridDim.x * blockDim.x)
        {
            float sum = 0.0f;
            for (int k = 0; k < colsA; ++k) 
            {
                sum += A[row * colsA + k] * B[k * colsB + col];
            }
            C[row * colsB + col] = sum;
        }
    }
}

// Funzione di controllo della correttezza
void checkCorrectness(float* A, float* B, float* C, int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < colsA; ++k) {
                sum += A[i * colsA + k] * B[k * colsB + j];
            }
            if (fabs(C[i * colsB + j] - sum) > 1e-5) {
                printf("Errore di correttezza nella posizione (%d, %d)\n", i, j);
                return;
            }
        }
    }
    printf("Controllo di correttezza superato!\n");
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Uso: %s <block_size_x> <block_size_y>\n", argv[0]);
        return 1;
    }

    int blockSizeX = atoi(argv[1]);
    int blockSizeY = atoi(argv[2]);

    float *A, *B, *C;
    float *d_A, *d_B, *d_C;
    
    // Allocazione della memoria sull'host
    A = (float*)malloc(ROWS_A * COLS_A * sizeof(float));
    B = (float*)malloc(ROWS_B * COLS_B * sizeof(float));
    C = (float*)malloc(ROWS_A * COLS_B * sizeof(float));
    
    // Allocazione della memoria sul device
    cudaMalloc(&d_A, ROWS_A * COLS_A * sizeof(float));
    cudaMalloc(&d_B, ROWS_B * COLS_B * sizeof(float));
    cudaMalloc(&d_C, ROWS_A * COLS_B * sizeof(float));
    
    // Inizializzazione delle matrici
    dim3 initBlockSize(32, 32);
    dim3 initGridSizeA((COLS_A + initBlockSize.x - 1) / initBlockSize.x, 
                       (ROWS_A + initBlockSize.y - 1) / initBlockSize.y);
    dim3 initGridSizeB((COLS_B + initBlockSize.x - 1) / initBlockSize.x, 
                       (ROWS_B + initBlockSize.y - 1) / initBlockSize.y);
    
    matrixInit<<<initGridSizeA, initBlockSize>>>(d_A, ROWS_A, COLS_A, 0.5f * powf(2, -10));
    matrixInit<<<initGridSizeB, initBlockSize>>>(d_B, ROWS_B, COLS_B, 2.0f);
    matrixInit<<<initGridSizeA, initBlockSize>>>(d_C, ROWS_A, COLS_B, 0.0f);
    
    // Configurazione del blocco in base all'input
    dim3 blockSize(blockSizeX, blockSizeY);
    dim3 gridSize((COLS_B + blockSize.x - 1) / blockSize.x, 
                  (ROWS_A + blockSize.y - 1) / blockSize.y);
    
    // Timer start per la versione monolitica
    auto start = std::chrono::high_resolution_clock::now();
    
    // Versione monolitica
    matrixMultMonolithic<<<gridSize, blockSize>>>(d_A, d_B, d_C, ROWS_A, COLS_A, COLS_B);
    cudaDeviceSynchronize();
    
    // Timer stop per la versione monolitica
    auto end = std::chrono::high_resolution_clock::now();
    auto monolithicTime = std::chrono::duration<float, std::milli>(end - start).count();
    printf("Monolithic,%dx%d,%.5f\n", blockSize.x, blockSize.y, monolithicTime);
    
    // Timer start per la versione grid-stride
    start = std::chrono::high_resolution_clock::now();
    
    // Versione grid-stride
    matrixMultGridStride<<<gridSize, blockSize>>>(d_A, d_B, d_C, ROWS_A, COLS_A, COLS_B);
    cudaDeviceSynchronize();
    
    // Timer stop per la versione grid-stride
    end = std::chrono::high_resolution_clock::now();
    auto gridStrideTime = std::chrono::duration<float, std::milli>(end - start).count();
    printf("GridStride,%dx%d,%.5f\n", blockSize.x, blockSize.y, gridStrideTime);
    
    // Copia del risultato sul host per il controllo
    cudaMemcpy(A, d_A, ROWS_A * COLS_A * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(B, d_B, ROWS_B * COLS_B * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(C, d_C, ROWS_A * COLS_B * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Controllo della correttezza
    checkCorrectness(A, B, C, ROWS_A, COLS_A, COLS_B);
    
    // Liberazione della memoria
    free(A); free(B); free(C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    return 0;
}