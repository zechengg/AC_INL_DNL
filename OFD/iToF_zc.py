import numpy as np

# ====== 系统参数 ======
v0 = 1800                # 电容初始电压 (mV)
pulse_interval = 10      # 激光脉冲间隔 (ns)
total_time = 500         # 总测距窗口长度 (ns)

initial_dv = 20          # 每个光子在 t=0 时刻的电压下降量 (mV)
total_dv_drop = 10       # 整个窗口内 ΔV 的最大下降量 (mV)

time_of_flight = 359.58    # 目标往返飞行时间 (ns)
jitter_sigma = 500       # 到达时间抖动标准差 (ps)
# ======================

# ====== 电压下降函数 ======
def voltage_drop(t):
    """返回在时间 t(ns) 时单个光子引起的电压下降量"""
    return initial_dv - total_dv_drop * (t / total_time)

# ====== TOF估计函数 ======
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
            #return round(t0, 3)

    return None

# ====== 主程序逻辑 ======

# 激光发射时间点
pulse_times = np.arange(0, total_time, pulse_interval)

# 光子到达时间 = 激光发射时间 + TOF
photon_arrival_times = pulse_times + time_of_flight

# 过滤出落在 [0, total_time] 范围内的光子
valid_indices = photon_arrival_times <= total_time
valid_arrival_times = photon_arrival_times[valid_indices]
num_valid_photons = len(valid_arrival_times)

# 添加抖动（正态分布，单位转换为 ns）
jitter_data = np.random.normal(loc=0, scale=jitter_sigma, size=num_valid_photons)
jittered_arrival_times = valid_arrival_times + jitter_data * 1e-3

# 模拟电压下降
voltage = v0
voltage_drops = []

for t in jittered_arrival_times:
    drop = voltage_drop(t)
    voltage -= drop
    voltage_drops.append(drop)

# ====== 输出信息 ======
print(f"\nTarget round-trip time: {time_of_flight} ns")
print(f"Number of photons arriving within {total_time}ns: {num_valid_photons}")
print("\nOriginal arrival times (ns):", valid_arrival_times)
print(f"\nInitial capacitor voltage: {v0} mV")
print(f"Voltage drops per photon (mV): {np.array(voltage_drops)}")
print(f"Final capacitor voltage after {total_time}ns: {voltage:.3f} mV")

# ====== TOF反推 ======
drop = v0 - voltage
#drop = round(v0 - voltage)
estimate_TOF = estimate_tof_from_voltage_drop(drop)
print("\ntime_of_flight =", time_of_flight)
print(f"ΔV={drop:.3f} mV → TOF =", estimate_TOF)