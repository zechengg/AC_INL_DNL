import numpy as np
import matplotlib.pyplot as plt

# 参数设置
initial_voltage = 1800     # 初始电压 (mV)
period = 10                # 放电周期 (ns)
total_time = 200           # 总测量时间 (ns)
jitter_sigma_ps = 200      # 抖动标准差（单位：皮秒）
initial_deltaV = 20
final_deltaV = 20
# 电压下降幅值随时间线性变化：t=0 时为20mV，t=400ns 时为10mV
def get_drop_amount(t):
    slope = (final_deltaV - initial_deltaV) / total_time  # 斜率
    return initial_deltaV + slope * t

# 初始化 start_time 列表：从 10 ns 到 400 ns，共10个点
start_times = np.linspace(20, 40, 253)

# 存储每个 start_time 的最终电压
final_voltages = []

# 对每个 start_time 进行一次模拟
for start_time in start_times:
    voltage = initial_voltage
    current_time = start_time

    while current_time <= total_time:
        drop = get_drop_amount(current_time)
        voltage -= drop

        # 添加抖动（将 ps 转换为 ns）
        jitter_ps = np.random.normal(loc=0, scale=jitter_sigma_ps)
        jitter_ns = jitter_ps / 1000.0  # 1 ns = 1000 ps

        # 更新下一次放电时间
        next_time = current_time + period + jitter_ns
        current_time = next_time

    final_voltages.append(voltage)
    print(f"start_time = {start_time:.2f} ns → 最终电压: {voltage:.2f} mV")

# 绘图展示
plt.figure(figsize=(10, 6))
plt.plot(start_times, final_voltages, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Start Time (ns)', fontsize=12)
plt.ylabel('Final Voltage (mV)', fontsize=12)
plt.title('Final Voltage vs Start Time', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()