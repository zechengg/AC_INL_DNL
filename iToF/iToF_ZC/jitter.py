import numpy as np
import matplotlib.pyplot as plt

# 假设 jitter_sigma_ps = 200
jitter_sigma_ps = 200
# 模拟生成 all_jitters_ns 数据
np.random.seed(0)  # 为了结果可复现
all_jitters_ns = np.random.normal(loc=0, scale=0.2, size=1000)


# 绘制直方图
plt.figure(figsize=(10, 5))
plt.hist(all_jitters_ns, bins=50, color='blue', alpha=0.7)
plt.title('Jitter Distribution (ns)')
plt.xlabel('Jitter (ns)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 绘制散点图（如果需要查看每个时间步的抖动情况）
plt.figure(figsize=(10, 5))
plt.scatter(range(len(all_jitters_ns)), all_jitters_ns, alpha=0.5, color='red')
plt.title('Jitter vs Time Step')
plt.xlabel('Time Step')
plt.ylabel('Jitter (ns)')
plt.grid(True)
plt.show()