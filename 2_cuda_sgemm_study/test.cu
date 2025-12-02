// # include <iostream>

// int main(){
//     std::cout << "hello cuda" << std::endl;
//     return 0;
// }
// minimal_kernel_test.cu
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void simple_kernel(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = idx * 2.0f;  // 简单计算
    
    if (idx == 0) {
        printf("🎯 核函数执行成功！线程0完成工作\n");
    }
}

int main() {
    printf("=== CUDA 核函数最小化测试 ===\n");
    
    // 1. 分配设备内存
    float* d_data;
    size_t size = 10 * sizeof(float);
    cudaMalloc(&d_data, size);
    printf("✅ 设备内存分配成功: %p\n", (void*)d_data);
    
    // 2. 启动核函数
    printf("启动核函数...\n");
    simple_kernel<<<1, 10>>>(d_data);  // 1个块，10个线程
    
    // 3. 检查启动错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("❌ 启动失败: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return -1;
    }
    printf("✅ 核函数启动成功\n");
    
    // 4. 同步检查执行错误
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("❌ 执行失败: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return -1;
    }
    printf("✅ 核函数执行完成\n");
    
    // 5. 验证结果
    float h_data[10];
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    
    printf("结果验证: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_data[i]);
    }
    printf("\n");
    
    cudaFree(d_data);
    printf("🎉 测试完成！\n");
    return 0;
}