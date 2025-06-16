import numpy as np
import matplotlib.pyplot as plt

# 参数设置
initial_voltage = 1800  # 初始电压 (mV)
period = 10  # 放电周期 (ns)
total_time = 500  # 总测量时间 (ns)
jitter_sigma_ps = 200  # 抖动标准差（单位：皮秒）
initial_deltaV = 30  # 初始下降幅值 (mV)
final_deltaV = 10  # 最终下降幅值 (mV)
n = 80  # 每个 start_time 模拟 n 次


# 获取当前周期内的 drop 幅值（线性变化）
def get_drop_amount(t):
    slope = (final_deltaV - initial_deltaV) / total_time
    return initial_deltaV + slope * t


# 初始化 start_time 列表
start_times = np.linspace(25, 400, 230)

# 存储每个 start_time 的统计结果
estimated_means = []
errors = []
std_devs = []

# 读取 zc_time_voltage_data.txt 文件
data = np.loadtxt('zc_time_voltage_data.txt')
zc_voltages = data[:, 1]  # 第二列是电压
zc_times = data[:, 0]  # 第一列是时间

# 对每个 start_time 进行 n 次模拟
for start_time in start_times:
    temp_estimated_times = []

    for _ in range(n):
        voltage = initial_voltage
        current_time = start_time

        while current_time <= total_time:
            drop = get_drop_amount(current_time)
            voltage -= drop

            # 添加抖动（将 ps 转换为 ns）
            jitter_ps = np.random.normal(loc=0, scale=jitter_sigma_ps)
            jitter_ns = jitter_ps / 1000.0

            # 更新下一次放电时间
            next_time = current_time + period + jitter_ns
            current_time = next_time

        # 找到最接近的 x 值（估计时间）
        closest_idx = np.abs(zc_voltages - round(voltage)).argmin()
        estimated_time = zc_times[closest_idx]
        temp_estimated_times.append(estimated_time)

    # 计算该 start_time 下的平均估计时间和误差
    avg_estimated = np.mean(temp_estimated_times)
    error = avg_estimated - start_time
    std_dev = np.std(temp_estimated_times)

    estimated_means.append(avg_estimated)
    errors.append(error)
    std_devs.append(std_dev)

# 输出整体统计信息（可选）
print(f"全局平均误差: {np.mean(errors):.2f} ns")
print(f"全局标准差: {np.mean(std_devs):.2f} ns")

# 绘图展示：误差和标准差随 start_time 变化
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# 图1：误差（Bias）
ax1.plot(start_times, errors, 'b.-', markersize=6, label='Error')
ax1.axhline(0, color='black', linestyle='--', linewidth=1)
ax1.set_ylabel('Error (Estimated - True) [ns]')
ax1.set_title('Error and Uncertainty vs Start Time', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

# 图2：标准差（Uncertainty）
ax2.plot(start_times, std_devs, 'g.-', markersize=6, label='Std Dev')
ax2.set_xlabel('Start Time (ns)')
ax2.set_ylabel('Standard Deviation [ns]')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.show()