import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ====== 参数设置 ======
v0 = 1800                # 电容初始电压 (mV)
pulse_interval = 10      # 激光脉冲间隔 (ns)
total_time = 500         # 总测距窗口长度 (ns)

initial_dv = 20          # 每个光子在 t=0 时刻的电压下降量 (mV)
total_dv_drop = 10       # 整个窗口内 ΔV 的最大下降量 (mV)

time_of_flight = 359.58  # 目标往返飞行时间 (ns)
jitter_sigma = 500       # 到达时间抖动标准差 (ps)

# ====== 正向模型：给定 TOF，计算 ΔV_total ======
def calculate_deltaV_from_tof(tof, T=pulse_interval, W=total_time,
                              V0=initial_dv, V_drop=total_dv_drop):
    slope = V_drop / W
    pulse_times = np.arange(0, W, T)
    valid_photons = pulse_times + tof <= W
    N = np.sum(valid_photons)

    if N == 0:
        return 0

    deltaV_total = N * V0 - slope * (N * tof + T * N * (N - 1) / 2)
    return deltaV_total

# ====== 反向模型：给定 ΔV，估算 TOF ======
def estimate_tof_from_voltage_drop(delta_v_mV, T=pulse_interval, W=total_time,
                                   V0=initial_dv, V_drop=total_dv_drop):
    slope_per_ns = V_drop / W
    max_N = (W // T) + 1
    epsilon = 1e-9

    for N in range(1, max_N + 1):
        t_min = max(0, W - N * T)
        t_max = W - (N - 1) * T

        if t_min >= t_max:
            continue

        try:
            numerator = N * V0 - delta_v_mV - slope_per_ns * T * N * (N - 1) / 2
            denominator = slope_per_ns * N
            t0 = numerator / denominator
        except ZeroDivisionError:
            continue

        if t_min - epsilon <= t0 <= t_max + epsilon:
            return t0

    return None

# ====== 生成 TOF → ΔV 数据 ======
tof_values = np.linspace(0, total_time, 500)
deltaV_values = [calculate_deltaV_from_tof(tof) for tof in tof_values]

# ====== 插值反函数：ΔV → TOF ======
# 去重并排序以避免重复 ΔV 导致插值错误
unique_indices = np.unique(deltaV_values, return_index=True)[1]
dv_unique = [deltaV_values[i] for i in unique_indices]
tof_unique = [tof_values[i] for i in unique_indices]

# 创建插值函数
func_tof_from_dv = interp1d(dv_unique, tof_unique, kind='linear', fill_value="extrapolate")

# 生成用于绘图的 ΔV 值
dv_plot_values = np.linspace(min(dv_unique), max(dv_unique), 500)
tof_plot_values = func_tof_from_dv(dv_plot_values)

# ====== 绘图：两个子图 ======
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# 图1：TOF → ΔV
axs[0].plot(tof_values, deltaV_values, color='blue')
axs[0].set_xlabel('TOF (ns)')
axs[0].set_ylabel('ΔV_total (mV)')
axs[0].set_title('TOF vs Total Voltage Drop (ΔV)')
axs[0].grid(True)

# 图2：ΔV → TOF（估计值）
axs[1].plot(dv_plot_values, tof_plot_values, color='orange')
axs[1].set_xlabel('ΔV_total (mV)')
axs[1].set_ylabel('Estimated TOF (ns)')
axs[1].set_title('ΔV vs Estimated TOF')
axs[1].grid(True)

plt.tight_layout()
plt.show()

# ====== 输出当前设置下的真实值 ======
real_dv = calculate_deltaV_from_tof(time_of_flight)
estimated_tof = estimate_tof_from_voltage_drop(real_dv)

print(f"\n真实 TOF: {time_of_flight} ns")
print(f"对应 ΔV: {real_dv:.3f} mV")
print(f"从 ΔV 反推得到 TOF: {estimated_tof:.3f} ns")