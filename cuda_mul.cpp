#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// CUDA Kernel
__global__ void matrixMul(int *A, int *B, int *C, int n) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < n && col < n) {

        int sum = 0;

        for(int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }

        C[row * n + col] = sum;
    }
}

int main() {

    int n;

    cout << "Enter order of square matrix: ";
    cin >> n;

    int size = n * n * sizeof(int);

    // Host matrices
    int *h_A = new int[n * n];
    int *h_B = new int[n * n];
    int *h_C = new int[n * n];

    cout << "Enter Matrix A:\n";

    for(int i = 0; i < n * n; i++)
        cin >> h_A[i];

    cout << "Enter Matrix B:\n";

    for(int i = 0; i < n * n; i++)
        cin >> h_B[i];

    // Device matrices
    int *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Block and Grid size
    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);

    // Kernel launch
    matrixMul<<<blocks, threads>>>(d_A, d_B, d_C, n);

    // Copy result back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cout << "\nResult Matrix:\n";

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            cout << h_C[i * n + j] << " ";
        }
        cout << endl;
    }

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}