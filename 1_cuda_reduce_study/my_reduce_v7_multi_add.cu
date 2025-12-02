#include <iostream>
#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256

template<unsigned int NUM_PER_BLOCK, unsigned int NUM_PER_THREAD>
__global__ void reduce(float *d_input, float *d_output){
    volatile __shared__ float shared[THREAD_PER_BLOCK];

    float *input_begin = d_input + NUM_PER_BLOCK * blockIdx.x;
    shared[threadIdx.x] = 0;
    for(int i = 0; i < NUM_PER_THREAD; i++){
        shared[threadIdx.x] += input_begin[threadIdx.x + i * THREAD_PER_BLOCK];
    }
    __syncthreads();
    
    if(THREAD_PER_BLOCK >= 512){
        if(threadIdx.x < 256)
                shared[threadIdx.x] += shared[threadIdx.x + 256];
        __syncthreads();
    }
    if(THREAD_PER_BLOCK >= 256){
        if(threadIdx.x < 128)
                shared[threadIdx.x] += shared[threadIdx.x + 128];
        __syncthreads();
    }
    if(THREAD_PER_BLOCK >= 128){
        if(threadIdx.x < 64)
                shared[threadIdx.x] += shared[threadIdx.x + 64];
        __syncthreads();
    }
    
    int tid = threadIdx.x;
    if(threadIdx.x  < 32){
        shared[tid] += shared[tid + 32];
        shared[tid] += shared[tid + 16];
        shared[tid] += shared[tid + 8];
        shared[tid] += shared[tid + 4];
        shared[tid] += shared[tid + 2];
        shared[tid] += shared[tid + 1];
    }

    if(threadIdx.x == 0)
        d_output[blockIdx.x] = shared[0];
}
bool check(float *out,float *res,int n){
    for(int i=0;i<n;i++){
        if(abs(out[i] - res[i]) > 0.005)
            return false;
    }
    return true;
}
int main()
{
    const int N = 32 * 1024 * 1024;
    float *input=(float *)malloc(N*sizeof(float));
    float *d_input;
    cudaMalloc((void **)&d_input, N * sizeof(float));
 
    // int block_num=N / THREAD_PER_BLOCK / 2;
    constexpr int block_num = 1024;
    constexpr int num_per_block = N / block_num;
    constexpr int num_per_thread = num_per_block / THREAD_PER_BLOCK;
    float *output=(float *)malloc(block_num * sizeof(float));
    float *d_output;
    cudaMalloc((void **)&d_output, block_num * sizeof(float));
    float *result=(float *)malloc(block_num * sizeof(float));
    
    for(int i=0;i<N;i++){
        input[i] = 2.0 * (float)drand48() - 1.0;
    }
    // cpu calc
    for(int i=0;i<block_num;i++){
        float cur = 0;
        for(int j=0; j< num_per_block; j++){
            cur += input[i * num_per_block + j];
        }
        result[i] = cur;
    }    

    cudaMemcpy(d_input, input, N*sizeof(float),cudaMemcpyHostToDevice);

    dim3 Grid( block_num);
    dim3 Block( THREAD_PER_BLOCK);

    reduce<num_per_block, num_per_thread><<<Grid,Block>>>(d_input ,d_output);

    cudaMemcpy(output ,d_output ,block_num*sizeof(float),cudaMemcpyDeviceToHost);

    if(check(output, result, block_num))printf("the ans is right\n");
    else{
        printf("the ans is wrong\n");
        for(int i=0;i<block_num;i++){
            printf("%lf ",output[i]);
        }
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;

}