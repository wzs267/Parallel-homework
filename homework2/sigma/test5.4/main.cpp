// two_way_sum_test.cpp
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>

const int n = 2000000;
const int iterations = 1000;

// 两路链式算法实现
double two_way_sum(const double* a, int size) {
    double sum1 = 0.0, sum2 = 0.0;
    int i = 0;
    for (; i + 1 < size; i += 2) {
        sum1 += a[i];
        sum2 += a[i+1];
    }
    for (; i < size; ++i) {
        sum1 += a[i];
    }
    return sum1 + sum2;
}

// 生成测试数据
void generate_data(double* data) {
    for (int i = 0; i < n; ++i) {
        data[i] = i% 100;
    }
}

int main() {
    double* data = new double[n];
    generate_data(data);

    // 预热缓存
    volatile double warmup = two_way_sum(data, n);

    // 正式测试
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        volatile double result = two_way_sum(data, n);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "两路链式算法测试结果:\n";
    std::cout << "数据量: " << n << " 个元素\n";
    std::cout << "迭代次数: " << iterations << " 次\n";
    std::cout << "平均耗时: " << duration.count()/iterations << " 毫秒\n";

    delete[] data;
    return 0;
}
