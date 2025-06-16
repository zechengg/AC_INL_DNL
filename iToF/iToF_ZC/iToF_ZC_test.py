import numpy as np
import matplotlib.pyplot as plt
from iToF.iToF_ZC.iToF_ZC import period, total_time, simulate_voltage_decay
from iToF.Lib.BG_Counts import *


# 读取 zc_time_voltage_data.txt 文件
data = np.loadtxt('zc_time_voltage_data.txt')
zc_voltages = data[:, 1]  # 第二列是电压
zc_times = data[:, 0]  # 第一列是时间

# 设置 start_time 列表（你可以根据需要调整）
start_time_list = np.linspace(20, 400, 52)  # 多个起始时间
num_runs = 50  # 每个 start_time 跑多少次

# 存储误差和标准差
mean_errors_t = []
std_devs_t = []


for start_time in start_time_list:
    error_t = []

    for _ in range(num_runs):
            # 生成事件
        events = generate_pulse_events(start_time, period, total_time,0.2)
        bg_events = generate_BG_events(0, total_time)
        final_events = apply_dead_time(events + bg_events, 5)
        # 计算 t1 和 t2
        voltage = simulate_voltage_decay(final_events)
        closest_idx = np.abs(zc_voltages - np.round(voltage)).argmin()
        estimated_time = zc_times[closest_idx]

        error_t.append(abs(estimated_time - start_time))

    # 计算统计量
    mean_errors_t.append(np.mean(error_t))
    std_devs_t.append(np.std(error_t))

# 绘图准备
x = start_time_list

# 创建两个子图：一个显示平均误差，一个显示标准差
fig, axs = plt.subplots(2, 1, figsize=(9, 6))

# 平均误差
axs[0].plot(x, mean_errors_t, 'o-', label='t1 Mean Error')
axs[0].set_title('Mean Error vs Start Time')
axs[0].set_ylabel('Mean Error')
axs[0].grid(True)
axs[0].legend()

# 标准差
axs[1].plot(x, std_devs_t, 'o-', label='t1 Std Dev', color='orange')
axs[1].set_title('Standard Deviation vs Start Time')
axs[1].set_xlabel('Start Time (ns)')
axs[1].set_ylabel('Std Dev')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()