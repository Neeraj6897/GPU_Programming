#include<iostream>
#include<vector>
#include<numeric> //for iota (to fill sequential numbers)
using namespace std;

#define cudaCheckErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess){
        fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit (code);
    }
}

//CUDA Kernel for exclusing scan (Prefix Sum)
__global__ void naive_parallel_scan(int *g_odata, int *g_idata, int n){
    //size is passed during kernel launch i.e. 3rd argument
    extern __shared__ int temp[];

    int thid = threadIdx.x;
    int pout = 0, pin = 1;

    //This is exclusive scan, so right shift by 1 and first element is set to 0
    temp[pout*n + thid] = (thid > 0) ? g_idata[thid-1] : 0;
    __syncthreads();

    for (int offset = 1; offset < n; offset *= 2) {
        pout = 1 - pout; //swap for double buffer indices
        pin = 1 - pout;

        if (thid >= offset) {
            //Each thread (thid >= offset) sums its current value
            // (from the 'pin' buffer) with a value from 'offset' positions behind it
            // in the 'pin' buffer. The result is stored in the 'pout' buffer.
            temp[pout*n + thid] = temp[pin*n + thid] + temp[pin*n + thid - offset];
        }
        else{
            // Threads whose 'thid' is less than 'offset' are not yet summing;
            // they simply copy their value from the 'pin' buffer to the 'pout' buffer
            // to propagate it to the next iteration.
            temp[pout*n + thid] = temp[pin*n + thid];
        }
        __syncthreads();        
    }

    g_odata[thid] = temp[pout*n + thid];  // write output to global memory
}

int main() {
    const int N = 8;
    const int block_size = 8;

    // Host (CPU) memory pointers
    vector<int> h_idata(N);
    vector<int> h_odata(N);

    // Initialize input data with (1, 2, 3, 4, 5, 6, 7, 8)
    iota(h_idata.begin(), h_idata.end(), 1);

    std::cout << "Original Host Data: [";
    for (int i = 0; i < N; ++i) {
        std::cout << h_idata[i] << (i == N - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;

    // Device (GPU) memory pointers
    int *d_idata = nullptr;
    int *d_odata = nullptr;

    // Allocate memory on device (gpu)
    cudaCheckErrors(cudaMalloc(&d_idata, N * sizeof(int)));
    cudaCheckErrors(cudaMalloc(&d_odata, N * sizeof(int)));

    //Copying input data from host to device
    cudaCheckErrors(cudaMemcpy(d_idata, h_idata.data(), N*sizeof(int), cudaMemcpyHostToDevice));

    // For double buffer, memory needed is twice the size of N
    size_t sharedMemSize = 2 * N * sizeof(int);

    // Kernel launch
    naive_parallel_scan<<<1, block_size, sharedMemSize>>>(d_odata, d_idata, N);

    cudaCheckErrors(cudaGetLastError());
    cudaCheckErrors(cudaDeviceSynchronize());

    //copy results from device to host
    cudaCheckErrors(cudaMemcpy(h_odata.data(), d_odata, N*sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "Scanned Host Data:  [";
    for (int i = 0; i < N; ++i) {
        std::cout << h_odata[i] << (i == N - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;

    cudaFree(d_idata);
    cudaFree(d_odata);
    
    return 0;
}