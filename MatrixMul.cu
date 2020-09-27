#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <cuda_runtime.h>

#define N (1 << 12)
#define tile_size 32
#define block_size tile_size

void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

void fillArray(float *arr){
    for(int i = 0; i < N*N; i++){
        arr[i] = rand() % 100;
        //arr[i] = i;
    }
}

void seqMatrixMul(float *a1, float *a2, float *aout){
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            aout[i*N+j]=0.0;
            for(int k = 0; k < N; k++){
                aout[N*i+j] += a1[N*i+k]*a2[N*k+j];
            }
        }
    }
}

void wrongNumberCheck(float *a1, float *a2){
    int bad = 0;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            if(a1[N*i + j] != a2[N*i + j]){
                bad = bad + 1;
            }
        }
    }
    printf("Number of wrong multiplications = %i\n", bad);
}

int mulCheck(float *a1, float *a2){
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            if(a1[N*i + j] != a2[N*i + j]){
                printf("Matrix Multiplication Failed!\n");
                printf("index = %i \n", N*i + j);
                printf("expected = %f\nreceived = %f\n", a1[N*i + j], a2[N*i+j]);
                printf("Next element...\n");
                printf("expected = %f\nreceived = %f\n", a1[N*i + j+1], a2[N*i+j+1]);
                printf("Checking for number of wrong multiplications...\n");
                wrongNumberCheck(a1, a2);
                return 1;
            }
        }
    }
    printf("Matrix Multiplication Successful!\n");
    return 0;
}

int mulCheck2(float *a1, float *a2){
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            if(abs(a1[N*i + j] - a2[N*i + j]) > 1e-8){
                printf("Matrix Multiplication Failed!\n");
                printf("index = %i \n", N*i + j);
                printf("row = %i, col = %i\n", i, j);
                printf("expected = %f\nreceived = %f\n", a1[N*i + j], a2[N*i+j]);
                printf("Next element...\n");
                printf("expected = %f\nreceived = %f\n", a1[N*i + j+1], a2[N*i+j+1]);
                printf("Checking for number of wrong multiplications...\n");
                wrongNumberCheck(a1, a2);
                return 1;
            }
        }
    }
    printf("Matrix Multiplication Successful!\n");
    return 0;
}

__global__ void gpuMatMul(float *a1, float *a2, float *aout){
    __shared__ float A[tile_size][tile_size];
    __shared__ float B[tile_size][tile_size];

    int tc = threadIdx.x;
    int tr = threadIdx.y;
    int c = blockIdx.x*tile_size + threadIdx.x;
    int r = blockIdx.y*tile_size + threadIdx.y;

    float sum_val = 0;
    for(int i = 0; i < N; i += tile_size){
        A[tr][tc] = a1[N*r + i + tc];
        B[tr][tc] = a2[c + N*(i + tr)];
        __syncthreads();
        for(int j = 0; j < tile_size; j++){
            sum_val += A[tr][j]*B[j][tc];
        }
        __syncthreads();
    }
    aout[N*r + c] = sum_val;
}

int main(void){
    // Setup time variables
    float timecpu = 0;
    float timegpu = 0;
    float tpcpu = 0;
    float tpgpu = 0;
    cudaEvent_t launch_begin_seq, launch_end_seq;

    // Host variables
    float *h_arr1 = (float*)malloc(N*N*sizeof(float));
    float *h_arr2 = (float*)malloc(N*N*sizeof(float));
    float *h_out = (float*)malloc(N*N*sizeof(float));
    float *h_save = (float*)malloc(N*N*sizeof(float));

    //Device variables
    float *d_arr1, *d_arr2, *d_out;
    cudaMalloc((void**)&d_arr1, N*N*sizeof(float));
    cudaMalloc((void**)&d_arr2, N*N*sizeof(float));
    cudaMalloc((void**)&d_out, N*N*sizeof(float));

    // Check Memory Allocation
    if(h_arr1 == 0 || h_arr2 == 0 || h_out == 0 || h_save == 0 || d_arr1 == 0 || d_arr2 == 0 || d_out == 0){
        printf("Memory Allocation Failed!\n");
        return 1;
    }

    // Fill Array
    fillArray(h_arr1);
    fillArray(h_arr2);
    memset(h_out, 0, N*N*sizeof(float));
    memset(h_save, 0, N*N*sizeof(float));

    // Create time variables
    cudaEventCreate(&launch_begin_seq);
    cudaEventCreate(&launch_end_seq);  

    //Start CPU Transpose
    cudaEventRecord(launch_begin_seq,0);
    seqMatrixMul(h_arr1, h_arr2, h_save);
    cudaEventRecord(launch_end_seq,0);

    cudaEventSynchronize(launch_end_seq);
    
    cudaEventElapsedTime(&timecpu, launch_begin_seq, launch_end_seq);
    printf("CPU time: %f ms\n", timecpu);
    tpcpu = 1e-9*2*N/(timecpu*1e-3);
    printf("Throughput = %f Gflops/s\n\n", tpcpu);

    // Prep Grid and Block variables
    dim3 dimGrid(N/tile_size, N/tile_size, 1);
    dim3 dimBlock(tile_size, block_size, 1);

    // Prep device memory
    cudaMemset(d_arr1, 0, N*N*sizeof(float));
    cudaMemset(d_arr2, 0, N*N*sizeof(float));
    cudaMemcpy(d_arr1, h_arr1, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr2, h_arr2, N*N*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(d_out, 0, N*N*sizeof(float));

    // Create time variables
    cudaEventCreate(&launch_begin_seq);
    cudaEventCreate(&launch_end_seq);

    // Start global GPU multiplication
    cudaEventRecord(launch_begin_seq,0);
    gpuMatMul<<<dimGrid, dimBlock>>>(d_arr1, d_arr2, d_out);
    cudaEventRecord(launch_end_seq,0);

    cudaEventSynchronize(launch_end_seq);

    // Copy Memory back to Host
    cudaMemcpy(h_out, d_out, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    // Check For Cuda Errors
    checkCUDAError("gpuMatMul");

    if(mulCheck2(h_save, h_out) == 0){
        cudaEventElapsedTime(&timegpu, launch_begin_seq, launch_end_seq);
        printf("GPU time: %f ms\n", timegpu);
        tpgpu = 1e-9*2*N/(timegpu*1e-3);
        printf("Throughput = %f Gflops/s\n\n", tpgpu);
    }

    printf("Speed up = %f \n", timecpu/timegpu);
    printf("ratio = %f \n\n", tpgpu/tpcpu);

    printf("CSV output:\n");
    printf("%i,%i,%i,%f,%f,%f,%f,%f,%f", N, tile_size, block_size, timecpu, timegpu, tpcpu, tpgpu, timecpu/timegpu, tpgpu/tpcpu);

    free(h_arr1);
    free(h_arr2);
    free(h_out);
    free(h_save);

    cudaFree(d_arr1);
    cudaFree(d_arr2);
    cudaFree(d_out);

    return 0;
}
