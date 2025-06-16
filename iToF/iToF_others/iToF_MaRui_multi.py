import numpy as np
import matplotlib.pyplot as plt

# 参数设置
jitter_sigma = 0.2      # 抖动标准差 (ns)
num_samples_per_run = 100   # 每次仿真的窗口计数次数
num_runs_per_time = 80    # 每个 real_time 重复运行多少次
time_range = (0, 6)         # 时间范围 (ns)
num_bins = 6                # 分成 6 个窗口
real_times = np.linspace(1.5, 2.5, 11)  # 在 [1.5, 2.5] 中取 11 个 real_time 点

# 存储结果
results = {
    'real_time': [],
    'mean_Tof': [],
    'std_Tof': [],
    'true_error': []
}

# 外层循环：对每个 real_time 进行多次仿真
for real_time in real_times:
    Tof_list = []

    for run in range(num_runs_per_time):
        counts = [0] * num_bins

        # 内部采样过程
        for _ in range(num_samples_per_run):
            measured_time = real_time + np.random.normal(loc=0, scale=jitter_sigma)

            if time_range[0] <= measured_time < time_range[1]:
                bin_index = int((measured_time - time_range[0]) // (time_range[1] / num_bins))
                if 0 <= bin_index < num_bins:
                    counts[bin_index] += 1

        # 计算 nab 差值
        n13 = counts[0] - counts[2]
        n24 = counts[1] - counts[3]
        n56 = counts[4] - counts[5]

        # 避免除以零
        denominator = abs(n13) + abs(n24) + abs(n56)
        if denominator == 0:
            x = float('nan')
        else:
            x = (n13 + n56) / denominator

        Tof = 1.5 - x
        if not np.isnan(Tof):
            Tof_list.append(Tof)

    # 统计当前 real_time 下的结果
    mean_Tof = np.mean(Tof_list)
    std_Tof = np.std(Tof_list)
    true_error = mean_Tof - real_time

    results['real_time'].append(real_time)
    results['mean_Tof'].append(mean_Tof)
    results['std_Tof'].append(std_Tof)
    results['true_error'].append(true_error)

# 转换为 numpy 数组
real_time_array = np.array(results['real_time'])
true_error_array = np.array(results['true_error'])
std_Tof_array = np.array(results['std_Tof'])

# === 绘图部分：只保留两张图 ===
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# 图1：True Error vs Real Time
axs[0].plot(real_time_array, true_error_array, 'ro-', label='True Error')
axs[0].axhline(0, color='gray', linestyle='--')
axs[0].set_xlabel('Real Time (ns)')
axs[0].set_ylabel('Error (ns)')
axs[0].set_title('Bias: True Error (Tof - Real Time)')
axs[0].grid(True)

# 图2：Standard Deviation vs Real Time
axs[1].plot(real_time_array, std_Tof_array, 'b^-', label='Std of Tof')
axs[1].set_xlabel('Real Time (ns)')
axs[1].set_ylabel('Standard Deviation (ns)')
axs[1].set_title('Precision: Standard Deviation vs Real Time')
axs[1].grid(True)

plt.tight_layout()
plt.show()