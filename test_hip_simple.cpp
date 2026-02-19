#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    std::cout << "Testing HIP/ROCm GPU access..." << std::endl;
    
    int deviceCount = 0;
    hipError_t error = hipGetDeviceCount(&deviceCount);
    
    if (error != hipSuccess) {
        std::cerr << "ERROR: hipGetDeviceCount failed: " << hipGetErrorString(error) << std::endl;
        std::cerr << "Error code: " << error << std::endl;
        return 1;
    }
    
    std::cout << "SUCCESS: Found " << deviceCount << " GPU(s)" << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        hipDeviceProp_t prop;
        error = hipGetDeviceProperties(&prop, i);
        
        if (error != hipSuccess) {
            std::cerr << "ERROR: hipGetDeviceProperties failed for GPU " << i << std::endl;
            continue;
        }
        
        std::cout << "\nGPU " << i << ":" << std::endl;
        std::cout << "  Name: " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
        std::cout << "  Clock Rate: " << prop.clockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Warp Size: " << prop.warpSize << std::endl;
    }
    
    // Try to allocate memory on GPU
    std::cout << "\nTesting GPU memory allocation..." << std::endl;
    void* d_ptr = nullptr;
    size_t size = 1024 * 1024; // 1 MB
    error = hipMalloc(&d_ptr, size);
    
    if (error != hipSuccess) {
        std::cerr << "ERROR: hipMalloc failed: " << hipGetErrorString(error) << std::endl;
        return 1;
    }
    
    std::cout << "SUCCESS: Allocated 1 MB on GPU" << std::endl;
    
    hipFree(d_ptr);
    std::cout << "SUCCESS: Freed GPU memory" << std::endl;
    
    std::cout << "\nAll tests passed! ROCm is working correctly." << std::endl;
    return 0;
}
