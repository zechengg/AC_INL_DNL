import numpy as np

# 参数设置
real_time = 2.5       # 真实事件发生时间 (ns)
jitter_sigma = 0.2      # 抖动标准差 (ns)
num_samples = 100        # 总采样次数
time_range = (0, 6)     # 时间范围 (ns)
num_bins = 6            # 分成 6 个窗口

# 初始化计数器
counts = [0] * num_bins  # 用列表存储6个窗口的计数

# 每次采样：real_time + jitter
for _ in range(num_samples):
    measured_time = real_time + np.random.normal(loc=0, scale=jitter_sigma)

    # 判断属于哪个窗口（0~6 ns 均分为6份 → 每份1 ns）
    if time_range[0] <= measured_time < time_range[1]:
        bin_index = int((measured_time - time_range[0]) // (time_range[1] / num_bins))
        if 0 <= bin_index < num_bins:
            counts[bin_index] += 1

# 输出各窗口计数结果
print("各窗口计数结果：")
for i, count in enumerate(counts):
    print(f"窗口 {i+1} ({i}-{i+1} ns): {count} 次")

# 定义 nab
n12 = counts[0] - counts[1]
n13 = counts[0] - counts[2]
n24 = counts[1] - counts[3]
n56 = counts[4] - counts[5]

# 计算 x
x = (n13 + n56) / (abs(n13) + abs(n24) + abs(n56))
Tof = 1.5-x
# 输出中间变量和最终 x
print("\n--- 中间变量 ---")
print(f"n12 = count1 - count2 = {counts[0]} - {counts[1]} = {n12}")
print(f"n13 = count1 - count3 = {counts[0]} - {counts[2]} = {n13}")
print(f"n24 = count2 - count4 = {counts[1]} - {counts[3]} = {n24}")
print(f"n56 = count5 - count6 = {counts[4]} - {counts[5]} = {n56}")

print("\n--- 最终结果 ---")
print(f"x = (n13 + n56) / (|n13| + |n24| + |n56|) = {x:.4f}")
print(f"Tof = {Tof:.4f}")