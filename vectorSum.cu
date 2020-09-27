#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <cuda_runtime.h>

#define N (1 << 25)
#define blocksize 8

void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

void fillArray(float *arr){
    for(int i = 0; i < N; i++){
        arr[i] = 1;
        //arr[i] = rand() % 100;
    }
}

void seqSum(float *a, float *out){
    for(int i = 0; i < N; i++){
        (*out) += a[i];
    }
}

__global__ void gpuGlobalSum(float *a, float *out){
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    int id = threadIdx.x;

    for(int k = blockDim.x/2; k > 0; k= k/2){
        if(id < k){
            a[i] += a[i + k];
        }
        __syncthreads();
    }
    if(id == 0){
        atomicAdd(out, a[i]);
    }
}

__global__ void gpuSharedSum(float *a, float *out){
    __shared__ float psum[blocksize];

    int i = threadIdx.x + blockDim.x*blockIdx.x;
    int id = threadIdx.x;

    psum[id] = (i < N) ? a[i]:0;
    __syncthreads();

    for(int k = blockDim.x/2; k > 0; k = k/2){
        if(id < k){
            psum[id] += psum[id + k];
        }
        __syncthreads();
    }
    if(id == 0){
        atomicAdd(out, psum[id]);
    }
}


int main(void){
    // Setup time variables
    float timecpu = 0;
    float timegpug = 0;
    float timegpus = 0;
    float tpcpu = 0;
    float tpgpug = 0;
    float tpgpus = 0;
    cudaEvent_t launch_begin_seq, launch_end_seq;

    // Host variables
    float *h_vec = (float*)malloc(N*sizeof(float));
    float h_result = 0.0;
    float *h_global = (float*)malloc(N*sizeof(float));
    float *h_shared = (float*)malloc(N*sizeof(float));

    //Device variables
    float *d_vec1, *d_vec2, *d_out1, *d_out2;
    cudaMalloc((void**)&d_vec1, N*sizeof(float));
    cudaMalloc((void**)&d_vec2, N*sizeof(float));
    cudaMalloc((void**)&d_out1, N*sizeof(float));
    cudaMalloc((void**)&d_out2, N*sizeof(float));

    // Check Memory Allocation
    if(h_vec == 0 || h_global == 0 || h_shared == 0 || d_vec1 == 0 || d_vec2 == 0 || d_out1 == 0 || d_out2 == 0){
        printf("Memory Allocation Failed!\n");
        return 1;
    }

    // Fill Array
    fillArray(h_vec);

    // Create time variables
    cudaEventCreate(&launch_begin_seq);
    cudaEventCreate(&launch_end_seq);  

    //Start CPU sum
    cudaEventRecord(launch_begin_seq,0);
    seqSum(h_vec, &h_result);
    cudaEventRecord(launch_end_seq,0);

    cudaEventSynchronize(launch_end_seq);
    
    cudaEventElapsedTime(&timecpu, launch_begin_seq, launch_end_seq);
    printf("CPU time: %f ms\n", timecpu);
    printf("Sum = %f\n\n", h_result);
    tpcpu = 1e-9*N/(timecpu*1e-3);
    printf("Throughput = %f Gflops/s\n\n", tpcpu);
    
    // Prep Grid and Block variables
    dim3 dimGrid(N/blocksize);
    dim3 dimBlock(blocksize);

    // Prep device memory
    cudaMemset(d_vec1, 0, N*sizeof(float));
    cudaMemcpy(d_vec1, h_vec, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_vec2, 0, N*sizeof(float));
    cudaMemcpy(d_vec2, h_vec, N*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(d_out1, 0, N*sizeof(float));
    cudaMemset(d_out2, 0, N*sizeof(float));

    // Create time variables
    cudaEventCreate(&launch_begin_seq);
    cudaEventCreate(&launch_end_seq);

    // Start global GPU sum
    cudaEventRecord(launch_begin_seq,0);
    gpuGlobalSum<<<dimGrid, dimBlock>>>(d_vec1, d_out1);
    cudaEventRecord(launch_end_seq,0);

    cudaEventSynchronize(launch_end_seq);

    // Copy Memory back to Host
    cudaMemcpy(h_global, d_out1, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Check For Cuda Errors
    checkCUDAError("gpuGlobalSum");

    if(h_result == h_global[0]){
        cudaEventElapsedTime(&timegpug, launch_begin_seq, launch_end_seq);
        printf("Sum Successful!\n");
        printf("Global Memory GPU time: %f ms\n\n", timegpug);
        tpgpug = 1e-9*N/(timegpug*1e-3);
        printf("Throughput = %f Gflops/s\n\n", tpgpug);
    }else{
        printf("Sum Failed!\n");
        printf("Expected = %f\nReceived = %f\n", h_result, h_global[0]);
    }

    // Create time variables
    cudaEventCreate(&launch_begin_seq);
    cudaEventCreate(&launch_end_seq);

    // Start shared GPU sum
    cudaEventRecord(launch_begin_seq,0);
    gpuSharedSum<<<dimGrid, dimBlock>>>(d_vec2, d_out2);
    cudaEventRecord(launch_end_seq,0);

    cudaEventSynchronize(launch_end_seq);

    // Copy Memory back to Host
    cudaMemcpy(h_shared, d_out2, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Check For Cuda Errors
    checkCUDAError("gpuSharedSum");

    if(h_result == h_shared[0]){
        cudaEventElapsedTime(&timegpus, launch_begin_seq, launch_end_seq);
        printf("Sum Successful!\n");
        printf("Shared Memory GPU time: %f ms\n\n", timegpus);
        tpgpus = 1e-9*N/(timegpus*1e-3);
        printf("Throughput = %f Gflops/s\n\n", tpgpus);
    }else{
        printf("Sum Failed!\n");
        printf("Expected = %f\nReceived = %f\n", h_result, h_shared[0]);
    }

    printf("Global Speed up = %f \n", timecpu/timegpug);
    printf("Global ratio = %f \n\n", tpgpug/tpcpu);

    printf("Shared Speed up = %f \n", timecpu/timegpus);
    printf("Shared ratio = %f \n\n", tpgpus/tpcpu);

    printf("CSV output:\n");
    printf("%i,%i,%i,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f", N, blocksize, blocksize, timecpu, timegpug, timegpus, tpcpu, tpgpug, tpgpus, timecpu/timegpug, timecpu/timegpus, tpgpug/tpcpu, tpgpus/tpcpu);

    free(h_vec);
    free(h_global);
    free(h_shared);

    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_out1);
    cudaFree(d_out2);

    return 0;
}
