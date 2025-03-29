// simple_sum_test.cpp
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>

const int n = 2000000;
const int iterations = 1000;

// 平凡算法实现
double simple_sum(const double* a, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        sum += a[i];
    }
    return sum;
}

// 生成测试数据
void generate_data(double* data) {
    for (int i = 0; i < n; ++i) {
        data[i] = i % 100;
    }
}

int main() {
    double* data = new double[n];
    generate_data(data);

    // 预热缓存
    volatile double warmup = simple_sum(data, n);

    // 正式测试
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        volatile double result = simple_sum(data, n);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "平凡算法测试结果:\n";
    std::cout << "数据量: " << n << " 个元素\n";
    std::cout << "迭代次数: " << iterations << " 次\n";
    std::cout << "平均耗时: " << duration.count()/iterations << " 毫秒\n";

    delete[] data;
    return 0;
}
