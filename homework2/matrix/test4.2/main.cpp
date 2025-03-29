#include <iostream>
#include <cstdlib>  // for rand()
#include <ctime>    // for time()
#include <chrono>   // for timing the execution

using namespace std;
using namespace std::chrono;

void cacheOptimizedMethod(int n) {
    double** B = new double*[n];
    for (int i = 0; i < n; ++i) {
        B[i] = new double[n];
    }
    double* a = new double[n];
    // 使用随机数填充矩阵 B 和向量 a
    for (int i = 0; i < n; ++i) {
        a[i] = i % 99;  // 随机生成 [0, 99] 范围内的值
        for (int j = 0; j < n; ++j) {
            B[i][j] = (i + j) % 99;  // 随机生成 [0, 99] 范围内的值
        }
    }

    double* sum = new double[n];
    for (int i = 0; i < n; ++i)
        sum[i] = 0.0;
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < n; ++row) {
            sum[row] += B[col][row] * a[col];
        }
    }
        for (int i = 0; i < n; ++i) {
        delete[] B[i];
    }
    delete[] B;
    delete[] a;
    delete[] sum;

}
int main() {
    auto start = high_resolution_clock::now();

    // 调用普通方法计算内积
    cacheOptimizedMethod(5000);
cacheOptimizedMethod(6000);
cacheOptimizedMethod(7000);
cacheOptimizedMethod(8000);
cacheOptimizedMethod(9000);
    auto end = high_resolution_clock::now();

    // 输出执行时间
    auto duration = duration_cast<milliseconds>(end - start);
    cout << "cache:Execution time: " << duration.count() << " milliseconds" << endl;


    cout << endl;

    // 释放内存

    return 0;
}

