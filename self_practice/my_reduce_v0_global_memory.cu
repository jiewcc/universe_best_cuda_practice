#include <iostream>

#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>
// #define CHECK(call)                                   \
// do                                                    \
// {                                                     \
//     const cudaError_t error_code = call;             \
//     if (error_code != cudaSuccess)                   \
//     {                                                 \
//         std::cout << "CUDA Error:" << std::endl;     \
//         std::cout << "    File: " << __FILE__ << std::endl; \
//         std::cout << "    Line: " << __LINE__ << std::endl; \
//         std::cout << "    Error code: " << error_code << std::endl; \
//         std::cout << "    Error info: " << cudaGetErrorString(error_code) << std::endl; \
//         exit(1);                                     \
//     }                                                 \
// } while (0)

int main()
{
    // printf("hello reduce\n");
    //   int dev = 0;
    // cudaDeviceProp devProp;
    // CHECK(cudaGetDeviceProperties(&devProp, dev));
    // std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
    // std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
    // std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    // std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
    // std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
    // std::cout << "每个SM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
 
    return 0;

}