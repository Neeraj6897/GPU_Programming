#include <iostream>
#include <cuda_runtime.h>

template <unsigned int blockSize>
__device__ void warpReduce(volatile int* sdata, int tid) {
    if (blockSize >= 64) { sdata[tid] = sdata[tid] + sdata[tid + 32]; }
    if (blockSize >= 32) { sdata[tid] = sdata[tid] + sdata[tid + 16]; }
    if (blockSize >= 16) { sdata[tid] = sdata[tid] + sdata[tid + 8]; }
    if (blockSize >= 8) { sdata[tid] = sdata[tid] + sdata[tid + 4]; }
    if (blockSize >= 4) { sdata[tid] = sdata[tid] + sdata[tid + 2]; }
    if (blockSize >= 2) { sdata[tid] = sdata[tid] + sdata[tid + 1]; }
}

// Kernel: Performs a simple tree-based reduction in shared memory.
template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];

    // Version 4: perform first reduction step during the global→shared load
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x*2) + threadIdx.x;
    // each thread loads two elements and immediately sums them
    sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
    __syncthreads();

    // Perform the reduction in shared memory.
    for (unsigned int s = blockDim.x/2; s > 32; s >>=1 ) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] = sdata[tid] + sdata[tid + 256];
        }
        __syncthreads();
    }

    if (blockSize >= 256) {
        if (tid < 128) {
            sdata[tid] = sdata[tid] + sdata[tid + 128];
        }
        __syncthreads();
    }

    if (blockSize >= 128) {
        if (tid < 64) {
            sdata[tid] = sdata[tid] + sdata[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warpReduce<blockSize>(sdata, tid);
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
    int numBlocks = (N + (2*blockSize) -1) / (2*blockSize);

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
    //Reduced the number of blocks to half (requirement for reduction4)
    switch (blockSize) {
        case 512: reduce6<512><<<numBlocks/2, blockSize, blockSize * sizeof(int)>>>(d_in, d_out); break;
        case 256: reduce6<256><<<numBlocks/2, blockSize, blockSize * sizeof(int)>>>(d_in, d_out); break;
        case 128: reduce6<128><<<numBlocks/2, blockSize, blockSize * sizeof(int)>>>(d_in, d_out); break;
        case 64: reduce6<64><<<numBlocks/2, blockSize, blockSize * sizeof(int)>>>(d_in, d_out); break;
        case 32: reduce6<32><<<numBlocks/2, blockSize, blockSize * sizeof(int)>>>(d_in, d_out); break;
        case 16: reduce6<16><<<numBlocks/2, blockSize, blockSize * sizeof(int)>>>(d_in, d_out); break;
        case 8: reduce6<8><<<numBlocks/2, blockSize, blockSize * sizeof(int)>>>(d_in, d_out); break;
        case 4: reduce6<4><<<numBlocks/2, blockSize, blockSize * sizeof(int)>>>(d_in, d_out); break;
        case 2: reduce6<2><<<numBlocks/2, blockSize, blockSize * sizeof(int)>>>(d_in, d_out); break;
        case 1: reduce6<1><<<numBlocks/2, blockSize, blockSize * sizeof(int)>>>(d_in, d_out); break;
    }
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
