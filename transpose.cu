#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N (1 << 12)
#define tile_size 64
#define block_size 16

void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

void serialTranspose(float *src, float *out){
    for(int i = 0; i < N*N; i++){
      int r = i / N;
      int c = i % N;
      int iT = N*c + r;
      out[iT] = src[i];
    }
}


int transposeCheck(float *src, float *out){
    for(int i = 0; i < N*N; i++){
      int r = i / N;
      int c = i % N;
      int iT = N*c + r;
      if(src[i] != out[iT]){
          printf("Transpose Incorrect\n");
          return 1;
      }
    }
    printf("Transpose Correct\n");
    return 0;
}

void fillArray(float *arr){
    for(int i = 0; i < N*N; i++){
        arr[i] = rand();
    }
}

__global__ void gpuTransposeGlobal(float *src, float *out){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < N*N){
        unsigned int iT = N*(i%N)+(i/N);
        out[iT] = src[i];
    }
}

__global__ void gpuTransposeShared(float *src, float *out){ 
    __shared__ float tile[tile_size][tile_size];

    int tc = threadIdx.x;
    int tr = threadIdx.y;
    int c = blockIdx.x*tile_size + threadIdx.x;
    int r = blockIdx.y*tile_size + threadIdx.y;

    for(int i = 0; i < tile_size; i = i + block_size){
        tile[tr+i][tc] = src[N*(r+i) + c];
    }
    __syncthreads();

    c = blockIdx.y*tile_size + threadIdx.x;
    r = blockIdx.x*tile_size + threadIdx.y;

    for(int i = 0; i < tile_size; i = i + block_size){
        out[N*(r+i)+c] = tile[tc][tr+i];
    }
}

int main(int argc, char** argv){
    // Setup time variables
    float timecpu = 0;
    float timegpug = 0;
    float timegpus = 0;
    float tpcpu = 0;
    float tpgpug = 0;
    float tpgpus = 0;
    cudaEvent_t launch_begin_seq, launch_end_seq;

    // Host variables
    float *h_arr = (float*)malloc(N*N*sizeof(float));
    float *h_out = (float*)malloc(N*N*sizeof(float));

    //Device variables
    float *d_arr, *d_out;
    cudaMalloc((void**)&d_arr, N*N*sizeof(float));
    cudaMalloc((void**)&d_out, N*N*sizeof(float));

    // Check Memory Allocation
    if(h_arr == 0 || h_out == 0 || d_arr == 0 || d_out == 0){
        printf("Memory Allocation Failed!\n");
        return 1;
    }

    // Fill Array
    fillArray(h_arr);
    memset(h_out, 0, N*N*sizeof(float));

    // Create time variables
    cudaEventCreate(&launch_begin_seq);
    cudaEventCreate(&launch_end_seq);

    //Start CPU Transpose
    cudaEventRecord(launch_begin_seq,0);
    serialTranspose(h_arr, h_out);
    cudaEventRecord(launch_end_seq,0);

    cudaEventSynchronize(launch_end_seq);
    
    if(transposeCheck(h_arr, h_out) == 0){
        cudaEventElapsedTime(&timecpu, launch_begin_seq, launch_end_seq);
        printf("CPU time: %f ms\n", timecpu);
        tpcpu = 1e-9*N*N/(timecpu*1e-3);
        printf("Throughput = %f Gflops/s\n\n", tpcpu);
    }

    // Prep Block And Thread variables
    size_t num_blocks = (N*N)/(block_size*tile_size);
    if((N*N) % block_size*tile_size) ++num_blocks; 

    // Prep device memory
    cudaMemset(d_arr, 0, N*N*sizeof(float));
    cudaMemcpy(d_arr, h_arr, N*N*sizeof(float), cudaMemcpyHostToDevice);

    memset(h_out, 0, N*N*sizeof(float));
    cudaMemset(d_out, 0, N*N*sizeof(float));

    // Create time variables
    cudaEventCreate(&launch_begin_seq);
    cudaEventCreate(&launch_end_seq);

    // Start global GPU Transpose
    cudaEventRecord(launch_begin_seq,0);
    gpuTransposeGlobal<<<num_blocks, block_size*tile_size>>>(d_arr, d_out);
    cudaEventRecord(launch_end_seq,0);

    cudaEventSynchronize(launch_end_seq);

    // Copy Memory back to Host
    cudaMemcpy(h_out, d_out, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    // Check For Cuda Errors
    checkCUDAError("gpuTranspose");

    if(transposeCheck(h_arr, h_out) == 0){
        cudaEventElapsedTime(&timegpug, launch_begin_seq, launch_end_seq);
        printf("Global Memory GPU time: %f ms\n", timegpug);
        tpgpug = 1e-9*N*N/(timegpug*1e-3);
        printf("Throughput = %f Gflops/s\n\n", tpgpug);
    }

    dim3 dimGrid(N/tile_size, N/tile_size, 1);
    dim3 dimBlock(tile_size, block_size, 1);

    memset(h_out, 0, N*N*sizeof(float));
    cudaMemset(d_out, 0, N*N*sizeof(float));

    // Create time variables
    cudaEventCreate(&launch_begin_seq);
    cudaEventCreate(&launch_end_seq);

    // Start shared GPU Transpose
    cudaEventRecord(launch_begin_seq,0);
    gpuTransposeShared<<<dimGrid, dimBlock>>>(d_arr, d_out);
    cudaEventRecord(launch_end_seq,0);

    cudaEventSynchronize(launch_end_seq);

    // Copy Memory back to Host
    cudaMemcpy(h_out, d_out, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    // Check For Cuda Errors
    checkCUDAError("gpuTransposeShared");

    if(transposeCheck(h_arr, h_out) == 0){
        cudaEventElapsedTime(&timegpus, launch_begin_seq, launch_end_seq);
        printf("Shared Memory GPU time: %f ms\n", timegpus);
        tpgpus = 1e-9*N*N/(timegpus*1e-3);
        printf("Throughput = %f Gflops/s\n\n", tpgpus);
    }

    printf("Global Speed up = %f \n", timecpu/timegpug);
    printf("Global ratio = %f \n\n", tpgpug/tpcpu);

    printf("Shared Speed up = %f \n", timecpu/timegpus);
    printf("Shared ratio = %f \n\n", tpgpus/tpcpu);

    printf("CSV output:\n");
    printf("%i,%i,%i,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f", N, tile_size, block_size, timecpu, timegpug, timegpus, tpcpu, tpgpug, tpgpus, timecpu/timegpug, timecpu/timegpus, tpgpug/tpcpu, tpgpus/tpcpu);

    // Free Host variables
    free(h_arr);
    free(h_out);

    // Free Device variables
    cudaFree(d_arr);
    cudaFree(d_out);
    return 0;
}
