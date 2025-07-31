import matplotlib.pyplot as plt
import numpy as np

# 数据准备
metrics = ['L1 Hit Rate', 'L2 Hit Rate', 'L3 Hit Rate', 'Memory Miss']
original = [95.02, 79.36, 60.90, 0.24]
optimized = [99.37, 89.80, 48.23, 0.037]
x = np.arange(len(metrics))

plt.figure(figsize=(10, 5))
bar_width = 0.35

# 绘制分组柱状图
plt.bar(x - bar_width/2, original, width=bar_width, label='Original', color='#FF9A76')
plt.bar(x + bar_width/2, optimized, width=bar_width, label='Optimized', color='#7FB77E')

# 图表装饰
plt.xticks(x, metrics)
plt.ylabel('Percentage (%)')
plt.title('Cache Performance Comparison')
plt.legend(loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加数值标签
for i in range(len(metrics)):
    plt.text(i - bar_width/2, original[i] + 1, f'{original[i]}%', ha='center')
    plt.text(i + bar_width/2, optimized[i] + 1, f'{optimized[i]}%', ha='center')

plt.savefig("cache_hitrate.png", dpi=300, bbox_inches='tight')
plt.close()