#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
#include <iostream>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
using namespace std;
using namespace chrono;

// 编译指令如下
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O1
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2
 //nvcc -DUSE_CUDA correctness.cpp train.cpp guessing.cpp guessing_cuda.cu md5.cpp -o main -O2 -std=c++11
double total_prepare = 0, total_kernel = 0, total_collect = 0;
int main()
{
    #ifdef USE_CUDA
    int device = 0;
    cudaSetDevice(device);
    #endif
    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("./input/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;


    // 加载一些测试数据
    unordered_set<std::string> test_set;
    ifstream test_data("./input/Rockyou-singleLined-full.txt");
    int test_count=0;
    string pw;
    while(test_data>>pw)
    {   
        test_count+=1;
        test_set.insert(pw);
        if (test_count>=1000000)
        {
            break;
        }
    }
    int cracked=0;

    q.init();
    cout << "here" << endl;
    int curr_num = 0;
    auto start = system_clock::now();
    int history = 0;
    while (!q.priority.empty())
    {
        q.PopNext();
        q.total_guesses = q.guesses.size();
        extern double last_prepare, last_kernel, last_collect;
        total_prepare += last_prepare;
        total_kernel += last_kernel;
        total_collect += last_collect;
        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "Guesses generated: " <<history + q.total_guesses << endl;
            curr_num = q.total_guesses;
            int generate_n=10000000;
            if (history + q.total_guesses > generate_n)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Guess time:" << time_guess - time_hash << "seconds"<< endl;
                cout << "Hash time:" << time_hash << "seconds"<<endl;
                cout << "Train time:" << time_train <<"seconds"<<endl;
                cout << "Total Prepare: " << total_prepare << "s, Total Kernel: " << total_kernel << "s, Total Collect: " << total_collect << "s\n";
                cout<<"Cracked:"<< cracked<<endl;
                break;
            }
        }
        if (curr_num > 1000000)
        {
            auto start_hash = system_clock::now();
            bit32 state[4];
            for (string pw : q.guesses)
            {
                if (test_set.find(pw) != test_set.end()) {
                    cracked+=1;
                }
                MD5Hash(pw, state);
            }
            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }
}
