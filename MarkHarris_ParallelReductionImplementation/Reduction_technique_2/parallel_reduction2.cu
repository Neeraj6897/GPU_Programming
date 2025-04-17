#include <iostream>
#include <cuda_runtime.h>

// Kernel: Performs a simple tree-based reduction in shared memory.
__global__ void reduce2(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];

    // Each thread loads one element from global memory into shared memory.
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();

    // Perform the reduction in shared memory.
    // In each iteration, threads with index divisible by (2*s) add the element "s" positions ahead.
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;

        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    // The first thread in each block writes the block's result to global memory.
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

int main() {
    // Total number of elements in the input array.
    int N = 134217728; // Make sure N is a multiple of blockSize for this simple example.
    int size = N * sizeof(int);

    // Set kernel launch parameters.
    int blockSize = 256;
    int numBlocks = N / blockSize; //(N + blockSize -1) / blockSize;

    // Allocate and initialize host memory.
    int *h_in  = new int[N];
    int *h_out = new int[numBlocks]; // One output value per block.
    for (int i = 0; i < N; i++) {
        h_in[i] = 1;  // Each element is 1, so final sum should equal N.
    }

    // Allocate device memory.
    int *d_in, *d_out;
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, numBlocks * sizeof(int));  // With blockSize 256.

    // Copy the input data from host to device.
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // Set kernel launch parameters.
    //int gridSize = N / blockSize; // Number of thread blocks.

    // Create CUDA events for timing.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event.
    cudaEventRecord(start, 0);

    // Launch the reduction kernel.
    // Third argument sets the shared memory size (in bytes) for each block.
    reduce2<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_in, d_out);

     // Record the stop event.
     cudaEventRecord(stop, 0);
     cudaEventSynchronize(stop);
 
     // Calculate elapsed time in milliseconds.
     float elapsedTime;
     cudaEventElapsedTime(&elapsedTime, start, stop);
     std::cout << "Kernel execution time: " << elapsedTime << " ms" << std::endl;
 
     // Clean up events.
     cudaEventDestroy(start);
     cudaEventDestroy(stop);

    // Copy the partial sums (one per block) back to host.
    cudaMemcpy(h_out, d_out, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    // Perform a final reduction on the host to sum up the partial results.
    int total_sum = 0;
    for (int i = 0; i < numBlocks; i++) {
        total_sum += h_out[i];
    }

    // Print the final result.
    std::cout << "Total sum is: " << total_sum << std::endl; // Should print 1024.

    // Free host and device memory.
    delete[] h_in;
    delete[] h_out;
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
