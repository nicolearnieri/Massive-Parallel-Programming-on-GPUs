#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>  // Aggiunto per i timer


#define ROWS_A (1 << 11)  // 2048 perché 2^11, shift a sinistra di 11 bit
#define COLS_A (1 << 10)  // 1024
#define ROWS_B COLS_A     // 1024 perché il numero di colonne della prima matrice deve essere uguale al numero di righe della seconda
#define COLS_B (1 << 9)   // 512


// Kernel per l'inizializzazione della matrice
__global__ void matrixInit(float* matrix, int rows, int cols, float value) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Calcolo dell'indice della colonna della matrice
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // Calcolo dell'indice della riga della matrice 
    //ricordare che riga e colonna in cuda sono invertite rispetto allo standard
    
    if (idx < cols && idy < rows)  // Controllo per evitare accessi fuori dalla matrice
    {
        matrix[idy * cols + idx] = value; // Inizializzazione dell'elemento della matrice con il valore passato come parametro
        //la matrice è memorizzata in modo lineare, quindi l'elemento di indice (i,j) è memorizzato in posizione i*cols+j
    }
}


// Kernel per la moltiplicazione delle matrici (versione monolitica)
__global__ void matrixMultMonolithic(float* A, float* B, float* C, int rowsA, int colsA, int colsB) 
{
    //calcolo degli indici
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rowsA && col < colsB) //controllo per non uscire dalla matrice
    {
        float sum = 0.0f; 
        for (int k = 0; k < colsA; ++k) 
        {
            sum += A[row * colsA + k] * B[k * colsB + col]; //moltiplicazione di ogni elemento della riga di A per la colonna di B
        }
        C[row * colsB + col] = sum; //salvataggio del risultato nella matrice C
    }
}


// Kernel per la moltiplicazione delle matrici (versione grid-stride loop)
__global__ void matrixMultGridStride(float* A, float* B, float* C, int rowsA, int colsA, int colsB) 
{
    for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < rowsA; row += gridDim.y * blockDim.y)  // Iterazione sulle righe
    {
        for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < colsB; col += gridDim.x * blockDim.x)  // Iterazione sulle colonne
        {
            float sum = 0.0f;
            for (int k = 0; k < colsA; ++k) 
            {
                sum += A[row * colsA + k] * B[k * colsB + col];
            }
            C[row * colsB + col] = sum; //salvataggio del risultato nella matrice C
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


int main() {
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
    
    // Configurazioni di blocco da testare
    int blockSizes[][2] = {{8,8}, {8,16}, {8,32}, {16,8}, {16,16}, {16,32}, {32,8}, {32,16}, {32,32}};
    
    for (int i = 0; i < 9; ++i) {
        dim3 blockSize(blockSizes[i][0], blockSizes[i][1]);
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
    }
    
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