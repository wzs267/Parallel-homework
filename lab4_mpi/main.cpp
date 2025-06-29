#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <mpi.h> // 引入MPI头文件，实现多进程并行
using namespace std;
using namespace chrono;

// 编译指令如下
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o test.exe
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o test.exe -O2
// g++ main.cpp train.cpp edition0.cpp md5.cpp -o test.exe -O2
// g++ main.cpp train.cpp edition1.cpp md5.cpp -o test.exe -O2
// g++ main.cpp train.cpp guessing_parallel.cpp md5.cpp -o test.exe -O2
// g++ lesson1.cpp -o test.exe -O2
//mpic++ main.cpp train.cpp guessing_mpi.cpp md5.cpp -o test_mpi -O2
// 主函数，支持MPI多进程并行猜测
// g++ main.cpp train.cpp guessing_mpi.cpp md5.cpp `
//   -I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" `
//   -L"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" `
//   -lmsmpi -o test_mpi.exe -O2
//mpiexec -n 4 test_mpi.exe

int main(int argc, char** argv)
{
    // 初始化MPI环境，获取当前进程编号(rank)和总进程数(size)
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;
    double total_send_time = 0.0;
double total_recv_time = 0.0;
double total_recv_mpi_time = 0.0;    // 纯MPI接收时间（前两个Recv）
double total_recv_process_time = 0.0;
double total_deserialize_time = 0.0;
double total_pushback_time = 0.0;
double worker_deserialize_PT_time = 0.0; // 处理接收数据的时间（内循环）
    PriorityQueue q;
    if (rank == 0) {
        // 主进程负责模型训练、任务分发和结果收集
        auto start_train = system_clock::now();
        q.m.train("./input/Rockyou-singleLined-full.txt"); // 训练模型
        q.m.order(); // 对模型数据排序
        auto end_train = system_clock::now();
        auto duration_train = duration_cast<microseconds>(end_train - start_train);
        time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;
        q.init(); // 初始化优先队列
        cout << "here" << endl;
        int curr_num = 0;
        auto start = system_clock::now();
        int history = 0;
        std::ofstream a("./output/results.txt");
        a << "start" << endl;
        // 优化：每轮分发batch_size个PT给每个worker，减少通信次数
        int batch_size = 20; // 可根据实际情况调整
        while (!q.priority.empty()) {
            int k = size - 1; // worker数量
            // 统计每个worker实际分配的PT数，生成发送的PT batch
            std::vector<std::vector<PT>> worker_batches(k);
            int pt_idx = 0;
            while (!q.priority.empty() && pt_idx < k * batch_size) {
                worker_batches[pt_idx % k].push_back(q.priority.front());

                    vector<PT> new_pts = q.priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        // 计算概率
        q.CalProb(pt);
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        for (auto iter = q.priority.begin(); iter != q.priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != q.priority.end() - 1 && iter != q.priority.begin())
            {
                // 判定概率
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    q.priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == q.priority.end() - 1)
            {
                q.priority.emplace_back(pt);
                break;
            }
            if (iter == q.priority.begin() && iter->prob < pt.prob)
            {
                q.priority.emplace(iter, pt);
                break;
            }
        }
    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
                q.priority.erase(q.priority.begin());
                pt_idx++;
            }
            // 发送每个worker的PT batch
            // auto start_send = system_clock::now();

for (int i = 0; i < k; ++i) {
    int batch_cnt = worker_batches[i].size();
    MPI_Send(&batch_cnt, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD);
    
    for (int j = 0; j < batch_cnt; ++j) {
        std::string pt_str = serialize_PT(worker_batches[i][j]);
        int len = pt_str.size();
        MPI_Send(&len, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD);
        MPI_Send(pt_str.data(), len, MPI_CHAR, i + 1, 0, MPI_COMM_WORLD);
    }
}

// auto end_send = system_clock::now();
// auto duration_send = duration_cast<microseconds>(end_send - start_send);
// total_send_time += double(duration_send.count()) * microseconds::period::num / microseconds::period::den;
            // 接收worker返回的所有guesses
            //auto start_recv = system_clock::now();
for (int i = 0; i < k; ++i) {
    // 计时开始：MPI接收部分
    //auto start_recv_mpi = system_clock::now();
    
    int total_guess_count = 0;
    MPI_Recv(&total_guess_count, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    std::vector<char> buf(total_guess_count * 64);
    MPI_Recv(buf.data(), total_guess_count * 64, MPI_CHAR, i + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    // auto end_recv_mpi = system_clock::now();
    // auto duration_recv_mpi = duration_cast<microseconds>(end_recv_mpi - start_recv_mpi);
    // total_recv_mpi_time += double(duration_recv_mpi.count()) * microseconds::period::num / microseconds::period::den;
    
// 计时开始：数据处理部分
auto start_recv_process = system_clock::now();

// 预计算所有字符串长度和总字符数
std::vector<size_t> lengths(total_guess_count);
size_t total_chars = 0;
for (int j = 0; j < total_guess_count; ++j) {
    //auto start_deserialize = system_clock::now();
    lengths[j] = strnlen(buf.data() + j * 64, 64);
    // auto end_deserialize = system_clock::now();
    // total_deserialize_time += duration_cast<nanoseconds>(end_deserialize - start_deserialize).count() / 1e9;
    total_chars += lengths[j];
}

// 优化push_back：直接resize后赋值
size_t old_size = q.guesses.size();
q.guesses.resize(old_size + total_guess_count);

// 并行处理字符串赋值
#pragma omp parallel for
for (int j = 0; j < total_guess_count; ++j) {
    // auto start_push = system_clock::now();
    q.guesses[old_size + j].assign(buf.data() + j * 64, lengths[j]);
    // auto end_push = system_clock::now();
    // total_pushback_time += duration_cast<nanoseconds>(end_push - start_push).count() / 1e9;
}

curr_num += total_guess_count;

// auto end_recv_process = system_clock::now();
// auto duration_recv_process = duration_cast<microseconds>(end_recv_process - start_recv_process);
// total_recv_process_time += double(duration_recv_process.count()) * microseconds::period::num / microseconds::period::den;
}

// auto end_recv = system_clock::now();
// auto duration_recv = duration_cast<microseconds>(end_recv - start_recv);
// total_recv_time += double(duration_recv.count()) * microseconds::period::num / microseconds::period::den;
            // 为了避免内存超限，定期对guesses做哈希处理
            if (curr_num > 1000000)
            {
                auto start_hash = system_clock::now();
                bit32 state[4];
                for (string pw : q.guesses)
                {
                    MD5Hash(pw, state); // 计算MD5哈希
                }
                auto end_hash = system_clock::now();
                auto duration = duration_cast<microseconds>(end_hash - start_hash);
                time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
                history += curr_num;
                curr_num = 0;
                q.guesses.clear();
            }
        
        if(history + curr_num > 30000000) {
                    // 通知所有worker进程退出
        for (int i = 1; i < size; ++i) {
            int batch_cnt = -1;
            MPI_Send(&batch_cnt, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
            auto end = system_clock::now();
            auto duration = duration_cast<microseconds>(end - start);
            time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
            cout << "Guess time:" << time_guess - time_hash << " seconds" << endl;
            cout << "Hash time:" << time_hash << " seconds" << endl;
            cout << "Train time:" << time_train << " seconds" << endl;
            // cout << "Total MPI send time: " << total_send_time << " seconds" << endl;
            // cout << "Total MPI receive time: " << total_recv_time << " seconds" << endl;
            // cout << "  Pure MPI Receive time: " << total_recv_mpi_time << " seconds" << endl;
            // cout << "  Data Processing time: " << total_recv_process_time << " seconds" << endl;
            // cout << "  Deserialization time: " << total_deserialize_time << " seconds" << endl;
            // cout << "  Pushback time: " << total_pushback_time << " seconds" << endl;
            // cout << history << " guesses generated." << endl;
            break;
        }
        }
        
    } else {
        // worker进程：每个进程独立初始化模型，循环接收PT batch并生成guesses
        q.m.train("./input/Rockyou-singleLined-full.txt");
        q.m.order();
        q.init();
        while (true) {
            int batch_cnt = 0;
            MPI_Recv(&batch_cnt, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (batch_cnt == -1) break;
            std::vector<std::string> all_guesses;
            for (int b = 0; b < batch_cnt; ++b) {
                int len = 0;
                MPI_Recv(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::vector<char> buf(len);
                MPI_Recv(buf.data(), len, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                //auto start_deserialize = system_clock::now();
                PT pt = deserialize_PT(std::string(buf.begin(), buf.end()));
                // auto end_deserialize = system_clock::now();
                // worker_deserialize_PT_time += duration_cast<nanoseconds>(end_deserialize - start_deserialize).count() / 1e9;

                std::vector<std::string> guesses;
                q.Generate(pt, guesses);
                all_guesses.insert(all_guesses.end(), guesses.begin(), guesses.end());
            }
            int total_guess_count = all_guesses.size();
            MPI_Send(&total_guess_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            std::vector<char> outbuf(total_guess_count * 64, 0);
            for (int i = 0; i < total_guess_count; ++i) {
                strncpy(outbuf.data() + i * 64, all_guesses[i].c_str(), 63);
            }
            MPI_Send(outbuf.data(), total_guess_count * 64, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
        //cout << "Worker " << rank << " finished processing." << endl;
        //cout << "Worker " << rank << " deserialization time: " << worker_deserialize_PT_time << " seconds" << endl;
    }
    MPI_Finalize(); // 结束MPI环境
    return 0;
}

// PT序列化为字符串，便于通过MPI发送
std::string serialize_PT(const PT& pt) {
    std::ostringstream oss;
    oss << pt.content.size() << " ";
    for (const auto& seg : pt.content) {
        oss << seg.type << " " << seg.length << " ";
    }
    oss << pt.pivot << " " << pt.preterm_prob << " " << pt.prob << " ";
    oss << pt.curr_indices.size() << " ";
    for (int idx : pt.curr_indices) oss << idx << " ";
    oss << pt.max_indices.size() << " ";
    for (int idx : pt.max_indices) oss << idx << " ";
    return oss.str();
}

// 字符串反序列化为PT，便于worker进程还原任务
PT deserialize_PT(const std::string& s) {
    std::istringstream iss(s);
    PT pt;
    int n;
    iss >> n;
    for (int i = 0; i < n; ++i) {
        int type, length;
        iss >> type >> length;
        pt.content.emplace_back(type, length);
    }
    iss >> pt.pivot >> pt.preterm_prob >> pt.prob;
    int sz;
    iss >> sz;
    pt.curr_indices.resize(sz);
    for (int i = 0; i < sz; ++i) iss >> pt.curr_indices[i];
    iss >> sz;
    pt.max_indices.resize(sz);
    for (int i = 0; i < sz; ++i) iss >> pt.max_indices[i];
    return pt;
}