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
__global__ void work_efficient_parallel_scan(int *g_odata, int *g_idata, int n){
    //size is passed during kernel launch i.e. 3rd argument
    extern __shared__ int temp[];

    int thid = threadIdx.x;
    int block_dim_x = blockDim.x; // No. of threads in the block

    // A. Load input into shared memory. Each thread loads two elements
    temp[2*thid] = g_idata[2*thid]; //will be used by thid for its 'left' value
    temp[2*thid + 1] = g_idata[2*thid + 1]; //will be used by thid for its 'right' value

    // B. Reduce Phase (Up-Sweep) - Building sums in place up the tree
    // Loop iterates from d = N/2, N/4, N/8...1
    int offset = 1;
    for(int d = n>>1; d>0; d >>= 1) // n >> 1 is n/2 and d >> 1 is d/2.
    {
        __syncthreads(); //to ensure that all threads have finished loading/prev writes
        if (thid < d)
        {
            //ai: Index of 'left' childs value in pair
            //bi: Index of 'right' childs value in pair
            int ai = offset * (2*thid + 1) - 1; //This refers to the element before bi in pair
            int bi = offset * (2*thid + 2) - 1; //This refers to the rightmost element of a pair

            temp[bi] = temp[bi] + temp[ai]; //temp[bi] holds the sum of temp[ai] and its original value
        }
        offset *= 2; //Double the offset for next iteration
    }
    //C. For exclusive scan, clear the root element i.e. last number to 0
    if (thid==0){
        temp[n-1] = 0;
    }

    //D. Down-Sweep Phase - Traversing down the tree and building the scan.
    //It distribures the sum collected in prev step and get the final scan value for each element
   for (int d=1; d<n; d *= 2){
    offset >>= 1; //Halsve the offset for next iteration
    __syncthreads();
    if (thid < d){
        int ai = offset * (2*thid + 1) -1;
        int bi = offset * (2*thid + 2) - 1;

        //Swap the values
        int t = temp[ai];
        temp[ai] = temp[bi]; //left child takes value of its parent which is at bi posn
        temp[bi] = temp[bi] + t; //right child takes temp[bi] plus original value of left child
    }
   }
   __syncthreads();

   //E. Write results from shraed memory to global device memory
   //Each threads writes its two calculated elements
   g_odata[2 * thid] = temp[2*thid];
   g_odata[2*thid + 1] = temp[2*thid + 1];
}

int main() {
    const int N = 8;
    const int block_size = N/2;

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
    size_t sharedMemSize = N * sizeof(int);

    // Kernel launch
    work_efficient_parallel_scan<<<1, block_size, sharedMemSize>>>(d_odata, d_idata, N);

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