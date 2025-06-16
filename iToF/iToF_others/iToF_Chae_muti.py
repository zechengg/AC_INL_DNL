import numpy as np
import matplotlib.pyplot as plt

# 参数设置
initial_voltage = 1800       # 初始电压 (mV)
period = 5                  # 理想周期长度 (ns)
total_time = 1000             # 主测量时间 (ns)
drop_per_period = 3          # 每次下降电压 (mV)
jitter_sigma_ps = 200        # 抖动标准差（单位：皮秒）
num_simulations = 100        # 每个 start_time 的仿真次数
start_time_list = np.linspace(10, 500, 42)  # 从 0 到 200 取 10 个 start_time

# 存储结果
results = {
    'start_time': [],
    'mean_t1': [],
    'std_t1': [],
    'mean_t2': [],
    'std_t2': []
}

# 模拟函数
def run_simulation(start_time):
    voltage = initial_voltage
    current_time = start_time  # 从 start_time 开始放电，无额外偏移

    NP0 = 0
    NP180 = 0
    NP90 = 0
    NP270 = 0

    while current_time <= total_time + start_time:
        voltage -= drop_per_period

        if start_time <= current_time <= total_time:
            NP0 += 1
            if current_time <= total_time / 2:
                NP90 += 1
            else:
                NP270 += 1
        elif total_time < current_time <= total_time + start_time:
            NP180 += 1
            NP270 += 1

        jitter_ps = np.random.normal(loc=0, scale=jitter_sigma_ps)
        jitter_ns = jitter_ps / 1000.0
        next_time = current_time + period + jitter_ns
        current_time = next_time

    try:
        alpha1 = NP180 / (NP0 + NP180)
    except ZeroDivisionError:
        alpha1 = float('nan')

    try:
        numerator = NP90 - NP270
        denominator = abs(NP0 - NP180) + abs(NP90 - NP270)
        alpha2 = -0.5 * numerator / denominator
    except ZeroDivisionError:
        alpha2 = float('nan')

    t1 = alpha1 * total_time if not np.isnan(alpha1) else np.nan
    t2 = alpha2 * total_time if not np.isnan(alpha2) else np.nan

    return t1, t2

# 主循环：对每个 start_time 进行多次仿真
for st in start_time_list:
    t1_values = []
    t2_values = []

    for sim in range(num_simulations):
        t1, t2 = run_simulation(st)
        if not np.isnan(t1):
            t1_values.append(t1)
        if not np.isnan(t2):
            t2_values.append(t2)

    mean_t1 = np.mean(t1_values) if len(t1_values) > 0 else np.nan
    std_t1 = np.std(t1_values) if len(t1_values) > 0 else np.nan
    mean_t2 = np.mean(t2_values) if len(t2_values) > 0 else np.nan
    std_t2 = np.std(t2_values) if len(t2_values) > 0 else np.nan

    results['start_time'].append(st)
    results['mean_t1'].append(mean_t1)
    results['std_t1'].append(std_t1)
    results['mean_t2'].append(mean_t2)
    results['std_t2'].append(std_t2)

# 转换为 numpy 数组
start_times = np.array(results['start_time'])
mean_t1 = np.array(results['mean_t1'])
error_t1 = mean_t1 - start_times
std_t1 = np.array(results['std_t1'])
mean_t2 = np.array(results['mean_t2'])
error_t2 = mean_t2 - start_times
std_t2 = np.array(results['std_t2'])
# 计算 mean 和 std 与 start_time 的比值
ratio_mean_t1 = 100 * (mean_t1 - start_times) / start_times
ratio_mean_t2 = 100 * (mean_t2 - start_times) / start_times

ratio_std_t1 = 100 * std_t1 / start_times
ratio_std_t2 = 100 * std_t2 / start_times
# 绘图：准确性 & 稳定性对比（已有）
plt.rcParams.update({
    'font.size': 18,  # 默认字体大小
    'axes.titlesize': 18,  # 子图标题字体大小
    'axes.labelsize': 18,  # 坐标轴标签字体大小
    'xtick.labelsize': 18,  # x轴刻度字体大小
    'ytick.labelsize': 18,  # y轴刻度字体大小
    'legend.fontsize': 18,  # 图例字体大小
    'figure.titlesize': 18  # 整个 figure 的标题字体大小（可选）
})
plt.figure(figsize=(12, 9))

plt.subplot(2, 2, 1)
plt.plot(start_times, error_t1, label='Error1', marker='o')
plt.plot(start_times, error_t2, label='Error2', marker='s')
plt.xlabel('Real Time (ns)')
plt.ylabel('Estimated error (ns)')

plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(start_times, std_t1, label='$\sigma1$', marker='o')
plt.plot(start_times, std_t2, label='$\sigma2$', marker='s')
plt.xlabel('Real Time (ns)')
plt.ylabel('Standard Deviation (ns)')

plt.legend()
plt.grid(True)

# 新增：比值分析图
plt.subplot(2, 2, 3)
plt.plot(start_times, ratio_mean_t1, label='Percentage Error1', marker='o', color='tab:blue')
plt.plot(start_times, ratio_mean_t2, label='Percentage Error2', marker='s', color='tab:orange')

plt.xlabel('Real Time (ns)')
plt.ylabel('Percentage Error(%)')

plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(start_times, ratio_std_t1, label='Percentage $\sigma$1', marker='o', color='tab:blue')
plt.plot(start_times, ratio_std_t2, label='Percentage $\sigma$2', marker='s', color='tab:orange')
plt.xlabel('Real Time (ns)')
plt.ylabel('Percentage $\sigma$(%)')

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()