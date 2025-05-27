#include "md5.h"
#include <iomanip>
#include <assert.h>
#include <chrono>
#include <arm_neon.h>
using namespace std;
using namespace chrono;

/**
 * StringProcess: 将单个输入字符串转换成MD5计算所需的消息数组
 * @param input 输入
 * @param[out] n_byte 用于给调用者传递额外的返回值，即最终Byte数组的长度
 * @return Byte消息数组
 */
Byte *StringProcess(string input, int *n_byte)
{
	//cout<<1<<endl;
	// if (input.empty()) {
    //     cout << "Error: Input string is empty!" << endl;
    // }
	// 将输入的字符串转换为Byte为单位的数组
	Byte *blocks = (Byte *)input.c_str();
	int length = input.length();
	//cout<<2<<endl;
	// 计算原始消息长度（以比特为单位）
	int bitLength = length * 8;

	// paddingBits: 原始消息需要的padding长度（以bit为单位）
	// 对于给定的消息，将其补齐至length%512==448为止
	// 需要注意的是，即便给定的消息满足length%512==448，也需要再pad 512bits
	int paddingBits = bitLength % 512;
	if (paddingBits > 448)
	{
		paddingBits += 512 - (paddingBits - 448);
	}
	else if (paddingBits < 448)
	{
		paddingBits = 448 - paddingBits;
	}
	else if (paddingBits == 448)
	{
		paddingBits = 512;
	}
	//cout<<3<<endl;
	// 原始消息需要的padding长度（以Byte为单位）
	int paddingBytes = paddingBits / 8;
	// 创建最终的字节数组
	// length + paddingBytes + 8:
	// 1. length为原始消息的长度（bits）
	// 2. paddingBytes为原始消息需要的padding长度（Bytes）
	// 3. 在pad到length%512==448之后，需要额外附加64bits的原始消息长度，即8个bytes
	int paddedLength = length + paddingBytes + 8;
	Byte *paddedMessage = new Byte[paddedLength];
	//cout<<4<<endl;
	// 复制原始消息
	memcpy(paddedMessage, blocks, length);

	// 添加填充字节。填充时，第一位为1，后面的所有位均为0。
	// 所以第一个byte是0x80
	paddedMessage[length] = 0x80;							 // 添加一个0x80字节
	memset(paddedMessage + length + 1, 0, paddingBytes - 1); // 填充0字节
	//cout<<5<<endl;
	// 添加消息长度（64比特，小端格式）
	for (int i = 0; i < 8; ++i)
	{
		// 特别注意此处应当将bitLength转换为uint64_t
		// 这里的length是原始消息的长度
		paddedMessage[length + paddingBytes + i] = ((uint64_t)length * 8 >> (i * 8)) & 0xFF;
	}
	//cout<<6<<endl;
	// 验证长度是否满足要求。此时长度应当是512bit的倍数
	int residual = 8 * paddedLength % 512;
	// assert(residual == 0);

	// 在填充+添加长度之后，消息被分为n_blocks个512bit的部分
	*n_byte = paddedLength;
	//cout<<7<<endl;
	return paddedMessage;
}


/**
 * MD5Hash: 将单个输入字符串转换成MD5
 * @param input 输入
 * @param[out] state 用于给调用者传递额外的返回值，即最终的缓冲区，也就是MD5的结果
 * @return Byte消息数组
 */
 void MD5Hash(string inputs[4], uint32_t state_neon[4][4]){
//cout<<"hash function start"<<endl;
	Byte **paddedMessage;
	//cout<<"	Byte **paddedMessage;"<<endl;	
	int *messageLength = new int[4];
	//cout<<"msglength=new int [4]"<<endl;
	paddedMessage = new Byte *[4]; 
	for (int i = 0; i < 4; i += 1)
	 {
//for (int i = 0; i < 4; ++i) {
	// 	cout << "Input[" << i << "] = " << inputs[i];
	// 	cout<<" ";
	// }
	// cout<<endl;
		paddedMessage[i] = StringProcess(inputs[i], &messageLength[i]);
// if (paddedMessage[i] == nullptr) {
//     cout << "Error: StringProcess returned nullptr for input[" << i << "]" << endl;
// }
// 		 cout<<"message lenth="<<messageLength[i]<<endl;
// 		  cout<<"Paddedmessage exist="<<(paddedMessage==nullptr)<<endl;
		//assert(messageLength[i] == messageLength[0]);//固定长度为64位的倍数,绝大多数都是64
	}
	int n_blocks = messageLength[0] / 64;
	//cout<<n_blocks<<endl;
//cout<<"paddedMessage generated"<<endl;
	// bit32* state= new bit32[4];
	uint32_t initial_state[4][4] = {
        {0x67452301, 0x67452301, 0x67452301, 0x67452301}, // a
        {0xEFCDAB89, 0xEFCDAB89, 0xEFCDAB89, 0xEFCDAB89}, // b
        {0x98BADCFE, 0x98BADCFE, 0x98BADCFE, 0x98BADCFE}, // c
        {0x10325476, 0x10325476, 0x10325476, 0x10325476}  // d
    };
    
    //将初始状态复制到输出状态
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            state_neon[i][j] = initial_state[i][j];
        }
    }
	//cout<<"state neon generated"<<endl;
	// 加载到 NEON 寄存器
	uint32x4_t a = vld1q_u32(state_neon[0]); // a[0]~a[3]
	uint32x4_t b = vld1q_u32(state_neon[1]); // b[0]~b[3]
	uint32x4_t c = vld1q_u32(state_neon[2]); // c[0]~c[3]
	uint32x4_t d = vld1q_u32(state_neon[3]);
	// 逐block地更新state
    for (int i = 0; i < n_blocks; i += 1)
    {
        // 并行生成 x[16]，每个x[i]是uint32x4_t（包含4个消息块的x[i]）
        uint32x4_t x_neon[16];
		for (int i1 = 0; i1 < 16; i1++) {
            // 获取4个消息块的基地址
            Byte* msg0 = &paddedMessage[0][i*64 + i1 * 4];
            Byte* msg1 = &paddedMessage[1][i*64 + i1 * 4];
            Byte* msg2 = &paddedMessage[2][i*64 + i1 * 4];
            Byte* msg3 = &paddedMessage[3][i*64 + i1 * 4];
            
            // 并行加载4个消息块的4字节
            uint32x4_t x0 = vmovq_n_u32(msg0[0] | (msg1[0] << 8) | (msg2[0] << 16) | (msg3[0] << 24));
            uint32x4_t x1 = vmovq_n_u32(msg0[1] | (msg1[1] << 8) | (msg2[1] << 16) | (msg3[1] << 24));
            uint32x4_t x2 = vmovq_n_u32(msg0[2] | (msg1[2] << 8) | (msg2[2] << 16) | (msg3[2] << 24));
            uint32x4_t x3 = vmovq_n_u32(msg0[3] | (msg1[3] << 8) | (msg2[3] << 16) | (msg3[3] << 24));
            
            // 组合结果
            x_neon[i1] = vorrq_u32(
                vshlq_n_u32(x3, 24),
                vorrq_u32(
                    vshlq_n_u32(x2, 16),
                    vorrq_u32(
                        vshlq_n_u32(x1, 8),
                        x0
                    )
                )
            );
        }
		//cout<<"x[i] generated"<<endl;
		auto start = system_clock::now();
		/* Round 1 */
FF_NEON(a, b, c, d, x_neon[0], vdupq_n_u32(s11), vdupq_n_u32(0xd76aa478));
FF_NEON(d, a, b, c, x_neon[1], vdupq_n_u32(s12), vdupq_n_u32(0xe8c7b756));
FF_NEON(c, d, a, b, x_neon[2], vdupq_n_u32(s13), vdupq_n_u32(0x242070db));
FF_NEON(b, c, d, a, x_neon[3], vdupq_n_u32(s14), vdupq_n_u32(0xc1bdceee));
FF_NEON(a, b, c, d, x_neon[4], vdupq_n_u32(s11), vdupq_n_u32(0xf57c0faf));
FF_NEON(d, a, b, c, x_neon[5], vdupq_n_u32(s12), vdupq_n_u32(0x4787c62a));
FF_NEON(c, d, a, b, x_neon[6], vdupq_n_u32(s13), vdupq_n_u32(0xa8304613));
FF_NEON(b, c, d, a, x_neon[7], vdupq_n_u32(s14), vdupq_n_u32(0xfd469501));
FF_NEON(a, b, c, d, x_neon[8], vdupq_n_u32(s11), vdupq_n_u32(0x698098d8));
FF_NEON(d, a, b, c, x_neon[9], vdupq_n_u32(s12), vdupq_n_u32(0x8b44f7af));
FF_NEON(c, d, a, b, x_neon[10], vdupq_n_u32(s13), vdupq_n_u32(0xffff5bb1));
FF_NEON(b, c, d, a, x_neon[11], vdupq_n_u32(s14), vdupq_n_u32(0x895cd7be));
FF_NEON(a, b, c, d, x_neon[12], vdupq_n_u32(s11), vdupq_n_u32(0x6b901122));
FF_NEON(d, a, b, c, x_neon[13], vdupq_n_u32(s12), vdupq_n_u32(0xfd987193));
FF_NEON(c, d, a, b, x_neon[14], vdupq_n_u32(s13), vdupq_n_u32(0xa679438e));
FF_NEON(b, c, d, a, x_neon[15], vdupq_n_u32(s14), vdupq_n_u32(0x49b40821));

/* Round 2 */
GG_NEON(a, b, c, d, x_neon[1], vdupq_n_u32(s21), vdupq_n_u32(0xf61e2562));
GG_NEON(d, a, b, c, x_neon[6], vdupq_n_u32(s22), vdupq_n_u32(0xc040b340));
GG_NEON(c, d, a, b, x_neon[11], vdupq_n_u32(s23), vdupq_n_u32(0x265e5a51));
GG_NEON(b, c, d, a, x_neon[0], vdupq_n_u32(s24), vdupq_n_u32(0xe9b6c7aa));
GG_NEON(a, b, c, d, x_neon[5], vdupq_n_u32(s21), vdupq_n_u32(0xd62f105d));
GG_NEON(d, a, b, c, x_neon[10], vdupq_n_u32(s22), vdupq_n_u32(0x02441453));
GG_NEON(c, d, a, b, x_neon[15], vdupq_n_u32(s23), vdupq_n_u32(0xd8a1e681));
GG_NEON(b, c, d, a, x_neon[4], vdupq_n_u32(s24), vdupq_n_u32(0xe7d3fbc8));
GG_NEON(a, b, c, d, x_neon[9], vdupq_n_u32(s21), vdupq_n_u32(0x21e1cde6));
GG_NEON(d, a, b, c, x_neon[14], vdupq_n_u32(s22), vdupq_n_u32(0xc33707d6));
GG_NEON(c, d, a, b, x_neon[3], vdupq_n_u32(s23), vdupq_n_u32(0xf4d50d87));
GG_NEON(b, c, d, a, x_neon[8], vdupq_n_u32(s24), vdupq_n_u32(0x455a14ed));
GG_NEON(a, b, c, d, x_neon[13], vdupq_n_u32(s21), vdupq_n_u32(0xa9e3e905));
GG_NEON(d, a, b, c, x_neon[2], vdupq_n_u32(s22), vdupq_n_u32(0xfcefa3f8));
GG_NEON(c, d, a, b, x_neon[7], vdupq_n_u32(s23), vdupq_n_u32(0x676f02d9));
GG_NEON(b, c, d, a, x_neon[12], vdupq_n_u32(s24), vdupq_n_u32(0x8d2a4c8a));

/* Round 3 */
HH_NEON(a, b, c, d, x_neon[5], vdupq_n_u32(s31), vdupq_n_u32(0xfffa3942));
HH_NEON(d, a, b, c, x_neon[8], vdupq_n_u32(s32), vdupq_n_u32(0x8771f681));
HH_NEON(c, d, a, b, x_neon[11], vdupq_n_u32(s33), vdupq_n_u32(0x6d9d6122));
HH_NEON(b, c, d, a, x_neon[14], vdupq_n_u32(s34), vdupq_n_u32(0xfde5380c));
HH_NEON(a, b, c, d, x_neon[1], vdupq_n_u32(s31), vdupq_n_u32(0xa4beea44));
HH_NEON(d, a, b, c, x_neon[4], vdupq_n_u32(s32), vdupq_n_u32(0x4bdecfa9));
HH_NEON(c, d, a, b, x_neon[7], vdupq_n_u32(s33), vdupq_n_u32(0xf6bb4b60));
HH_NEON(b, c, d, a, x_neon[10], vdupq_n_u32(s34), vdupq_n_u32(0xbebfbc70));
HH_NEON(a, b, c, d, x_neon[13], vdupq_n_u32(s31), vdupq_n_u32(0x289b7ec6));
HH_NEON(d, a, b, c, x_neon[0], vdupq_n_u32(s32), vdupq_n_u32(0xeaa127fa));
HH_NEON(c, d, a, b, x_neon[3], vdupq_n_u32(s33), vdupq_n_u32(0xd4ef3085));
HH_NEON(b, c, d, a, x_neon[6], vdupq_n_u32(s34), vdupq_n_u32(0x04881d05));
HH_NEON(a, b, c, d, x_neon[9], vdupq_n_u32(s31), vdupq_n_u32(0xd9d4d039));
HH_NEON(d, a, b, c, x_neon[12], vdupq_n_u32(s32), vdupq_n_u32(0xe6db99e5));
HH_NEON(c, d, a, b, x_neon[15], vdupq_n_u32(s33), vdupq_n_u32(0x1fa27cf8));
HH_NEON(b, c, d, a, x_neon[2], vdupq_n_u32(s34), vdupq_n_u32(0xc4ac5665));

/* Round 4 */
II_NEON(a, b, c, d, x_neon[0], vdupq_n_u32(s41), vdupq_n_u32(0xf4292244));
II_NEON(d, a, b, c, x_neon[7], vdupq_n_u32(s42), vdupq_n_u32(0x432aff97));
II_NEON(c, d, a, b, x_neon[14], vdupq_n_u32(s43), vdupq_n_u32(0xab9423a7));
II_NEON(b, c, d, a, x_neon[5], vdupq_n_u32(s44), vdupq_n_u32(0xfc93a039));
II_NEON(a, b, c, d, x_neon[12], vdupq_n_u32(s41), vdupq_n_u32(0x655b59c3));
II_NEON(d, a, b, c, x_neon[3], vdupq_n_u32(s42), vdupq_n_u32(0x8f0ccc92));
II_NEON(c, d, a, b, x_neon[10], vdupq_n_u32(s43), vdupq_n_u32(0xffeff47d));
II_NEON(b, c, d, a, x_neon[1], vdupq_n_u32(s44), vdupq_n_u32(0x85845dd1));
II_NEON(a, b, c, d, x_neon[8], vdupq_n_u32(s41), vdupq_n_u32(0x6fa87e4f));
II_NEON(d, a, b, c, x_neon[15], vdupq_n_u32(s42), vdupq_n_u32(0xfe2ce6e0));
II_NEON(c, d, a, b, x_neon[6], vdupq_n_u32(s43), vdupq_n_u32(0xa3014314));
II_NEON(b, c, d, a, x_neon[13], vdupq_n_u32(s44), vdupq_n_u32(0x4e0811a1));
II_NEON(a, b, c, d, x_neon[4], vdupq_n_u32(s41), vdupq_n_u32(0xf7537e82));
II_NEON(d, a, b, c, x_neon[11], vdupq_n_u32(s42), vdupq_n_u32(0xbd3af235));
II_NEON(c, d, a, b, x_neon[2], vdupq_n_u32(s43), vdupq_n_u32(0x2ad7d2bb));
II_NEON(b, c, d, a, x_neon[9], vdupq_n_u32(s44), vdupq_n_u32(0xeb86d391));

// 最终状态更新
a = vaddq_u32(a, vld1q_u32(&state_neon[0][0]));
b = vaddq_u32(b, vld1q_u32(&state_neon[1][0]));
c = vaddq_u32(c, vld1q_u32(&state_neon[2][0]));
d = vaddq_u32(d, vld1q_u32(&state_neon[3][0]));

// 存储结果
vst1q_u32(&state_neon[0][0], a);
vst1q_u32(&state_neon[1][0], b);
vst1q_u32(&state_neon[2][0], c);
vst1q_u32(&state_neon[3][0], d);
	}

	// 下面的处理，在理解上较为复杂
	for(int j=0;j<3;j++){
	for (int i = 0; i < 4; i++)
	{
		uint32_t value = state_neon[j][i];
		state_neon[j][i] = ((value & 0xff) << 24) |		 // 将最低字节移到最高位
				   ((value & 0xff00) << 8) |	 // 将次低字节左移
				   ((value & 0xff0000) >> 8) |	 // 将次高字节右移
				   ((value & 0xff000000) >> 24); // 将最高字节移到最低位
	}
	}
	// 输出最终的hash结果
	// for (int i1 = 0; i1 < 4; i1 += 1)
	// {
	// 	cout << std::setw(8) << std::setfill('0') << hex << state[i1];
	// }
	// cout << endl;// 存储结果
vst1q_u32(&state_neon[0][0], a);
vst1q_u32(&state_neon[1][0], b);
vst1q_u32(&state_neon[2][0], c);
vst1q_u32(&state_neon[3][0], d);

	// 释放动态分配的内存
	// 实现SIMD并行算法的时候，也请记得及时回收内存！
	delete[] paddedMessage;
	delete[] messageLength;
}