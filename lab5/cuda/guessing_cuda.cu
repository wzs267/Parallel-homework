#include <cuda_runtime.h>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
//代码逻辑：主函数传要加的字符串数组，长度，如果是第二个for还需传入前缀guess，(最后两个参数是循环控制符pt.max_indices[pt.content.size() - 1]和回传的大数组)在gpu中进行拼接并合成一个大字符串传回主函数。
// extern "C" void cuda_generate_guesses(
//     char **h_values, int value_len, char *h_guess_prefix, int prefix_len, int n, std::vector<std::string> &guesses);
__global__ void generate_guesses_kernel_flat(const char *flat_values, int value_len, const char *d_guess_prefix, int prefix_len, char *result_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        char *dst = result_data + idx * (prefix_len + value_len + 1);
        if (prefix_len > 0) {
            for (int i = 0; i < prefix_len; ++i) dst[i] = d_guess_prefix[i];
        }
        const char *src = flat_values + idx * (value_len + 1);
        for (int i = 0; i < value_len; ++i) dst[prefix_len + i] = src[i];
        dst[prefix_len + value_len] = '\0';
    }
}

// 新接口，flat_values直接传入，增加offset参数
extern "C" void cuda_generate_guesses(
    const char *flat_values, int value_len, char *h_guess_prefix, int prefix_len, int n, std::vector<std::string> &guesses, size_t offset,
    double* prepare_time, double* kernel_time, double* collect_time)
{
    using namespace std::chrono;
    //auto t_prepare0 = high_resolution_clock::now();
    // 1. flat_values已由主机端准备好，无需再构造
    size_t flat_size = n * (value_len + 1);
    // 2. 分配device内存
    char *d_flat_values, *d_guess_prefix = nullptr, *d_result_data;
    cudaMalloc(&d_flat_values, flat_size * sizeof(char));
    cudaMemcpy(d_flat_values, flat_values, flat_size * sizeof(char), cudaMemcpyHostToDevice);
    if (prefix_len > 0) {
        cudaMalloc(&d_guess_prefix, prefix_len * sizeof(char));
        cudaMemcpy(d_guess_prefix, h_guess_prefix, prefix_len * sizeof(char), cudaMemcpyHostToDevice);
    }
    size_t result_size = n * (prefix_len + value_len + 1);//结果字符串长度
    cudaMalloc(&d_result_data, result_size * sizeof(char));
    //auto t_prepare1 = high_resolution_clock::now();
    //if (prepare_time) *prepare_time = duration<double>(t_prepare1 - t_prepare0).count();
    // kernel
    //auto t_kernel0 = high_resolution_clock::now();
    int block = 256;
    int grid = (n + block - 1) / block;
    generate_guesses_kernel_flat<<<grid, block>>>(d_flat_values, value_len, d_guess_prefix, prefix_len, d_result_data, n);
    cudaDeviceSynchronize();
    //auto t_kernel1 = high_resolution_clock::now();
    //if (kernel_time) *kernel_time = duration<double>(t_kernel1 - t_kernel0).count();
    // collect
    // auto t_collect0 = high_resolution_clock::now();
    char *result_data = new char[result_size];
    cudaMemcpy(result_data, d_result_data, result_size * sizeof(char), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        guesses[offset + i].assign(result_data + i * (prefix_len + value_len + 1));
    }
    // auto t_collect1 = high_resolution_clock::now();
    // if (collect_time) *collect_time = duration<double>(t_collect1 - t_collect0).count();
    // // 释放内存
    delete[] result_data;
    cudaFree(d_flat_values);
    cudaFree(d_result_data);
    if (d_guess_prefix) cudaFree(d_guess_prefix);
}
