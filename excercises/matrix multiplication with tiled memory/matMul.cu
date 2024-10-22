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

__global__ void matrixMultTiled(float* A, float* B, float* C, int rowsA, int colsA, int colsB) 
{
    // Dimensioni del tile: uguale alla dimensione del blocco
    extern __shared__ float sharedMem[];
    float* shared_A = sharedMem;                            // Shared memory per il blocco di A
    float* shared_B = sharedMem + blockDim.x * blockDim.y;   // Shared memory per il blocco di B

    int row = blockIdx.y * blockDim.y + threadIdx.y;         // Indice della riga di C
    int col = blockIdx.x * blockDim.x + threadIdx.x;         // Indice della colonna di C
    float sum = 0.0f;                                        // Risultato parziale per C[row][col]

    // Ciclo sui sotto-blocchi (tiles) di A e B necessari per il calcolo di C
    for (int tileIdx = 0; tileIdx < (colsA + blockDim.x - 1) / blockDim.x; ++tileIdx) {
        // Carica un elemento del blocco di A nella shared memory
        if (row < rowsA && (tileIdx * blockDim.x + threadIdx.x) < colsA) {
            shared_A[threadIdx.y * blockDim.x + threadIdx.x] = A[row * colsA + tileIdx * blockDim.x + threadIdx.x];
        } else {
            shared_A[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;
        }

        // Carica un elemento del blocco di B nella shared memory
        if (col < colsB && (tileIdx * blockDim.y + threadIdx.y) < colsA) {
            shared_B[threadIdx.y * blockDim.x + threadIdx.x] = B[(tileIdx * blockDim.x + threadIdx.y) * colsB + col];
        } else {
            shared_B[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;
        }

        // Sincronizzazione dei thread per assicurarsi che tutti i dati del blocco siano caricati
        __syncthreads();

        // Moltiplicazione dei blocchi
        for (int k = 0; k < blockDim.x; ++k) {
            sum += shared_A[threadIdx.y * blockDim.x + k] * shared_B[k * blockDim.x + threadIdx.x];
        }

        // Sincronizzazione prima di caricare il prossimo blocco
        __syncthreads();
    }

    // Scrivi il risultato finale nella matrice C
    if (row < rowsA && col < colsB) {
        C[row * colsB + col] = sum;
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

    // Allocazione della shared memory (2 * blockSizeX * blockSizeY)
    size_t sharedMemSize = 2 * blockSizeX * blockSizeY * sizeof(float);

    // Timer start per la versione tiled
    auto start = std::chrono::high_resolution_clock::now();

    // Versione tiled
    matrixMultTiled<<<gridSize, blockSize, sharedMemSize>>>(d_A, d_B, d_C, ROWS_A, COLS_A, COLS_B);
    cudaDeviceSynchronize();

    // Timer stop per la versione tiled
    auto end = std::chrono::high_resolution_clock::now();
    auto tiledTime = std::chrono::duration<float, std::milli>(end - start).count();
    printf("Tiled,%dx%d,%.5f\n", blockSize.x, blockSize.y, tiledTime);
    
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
