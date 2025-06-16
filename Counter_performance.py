import numpy as np
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

# ====== 用户配置参数 ====== #
num_points_to_keep = 65
period = 10
integration_length = 8
start_index = 2  # 从第二个周期开始
total_length = (num_points_to_keep + start_index) * period

# ====== 数据读取与过滤 ====== #
raw_xs = []
raw_ys = []

with open('data.txt', 'r') as f:
    for line in f:
        if not line.strip():
            continue
        x_str, y_str = line.strip().split()
        x = float(x_str) * 1e9
        y = float(y_str)
        if x < total_length:
            raw_xs.append(x)
            raw_ys.append(y)

# 转为 NumPy 数组
xs = np.array(raw_xs)
ys = np.array(raw_ys)


# 存储结果
bin_means = []

# 构造周期边界：[0, 10), [10, 20), ..., [640, 650)
bin_edges = np.arange(0, total_length + period, period)


for i in range(start_index, len(bin_edges) - 1):
    start = bin_edges[i]
    end = bin_edges[i + 1]

    # 取出当前周期后 8 个单位范围内的数据
    mask = (xs >= start + (period - integration_length)) & (xs < end)
    x_sub = xs[mask]
    y_sub = ys[mask]

    if len(x_sub) < 2:
        bin_means.append(np.nan)
    else:
        integral = trapezoid(y_sub, x=x_sub)
        avg = integral /  (x_sub[-1] - x_sub[0])
        bin_means.append(avg)

# ====== 输出结果 ====== #
for i, mean in enumerate(bin_means):
    eff_start = bin_edges[i + start_index] + (period - integration_length)
    eff_end = bin_edges[i + start_index] + period
    print(f"区间 [{eff_start}, {eff_end}): 平均值 = {mean:.6f}")

# 计算 DNL 和 INL
# 假设 bin_means 是 ADC 输出的码字对应的电压值
# 计算 LSB
LSB = (max(bin_means) - min(bin_means)) / (len(bin_means) - 1)

# 初始化 DNL 和 INL 数组
DNL = []
INL = []
Voltage_steps = []
# 计算 DNL 和 INL
for i in range(1, len(bin_means) - 1):  # 从第二个到倒数第二个元素

    # 计算 DNL
    dnl = ((bin_means[i] - bin_means[i - 1]) / LSB) + 1
    v_step = bin_means[i] - bin_means[i - 1]
    DNL.append(dnl)
    Voltage_steps.append(v_step)

    # 计算 INL 使用 DNL 累积和
    inl = sum(DNL[:i])
    INL.append(inl)

# ====== 绘图部分 - 2x2 布局 ======

plt.rcParams.update({
    'font.size': 18,  # 默认字体大小
    'axes.titlesize': 18,  # 子图标题字体大小
    'axes.labelsize': 18,  # 坐标轴标签字体大小
    'xtick.labelsize': 18,  # x轴刻度字体大小
    'ytick.labelsize': 18,  # y轴刻度字体大小
    'legend.fontsize': 18,  # 图例字体大小
    'figure.titlesize': 18  # 整个 figure 的标题字体大小（可选）
})
fig, axs = plt.subplots(2, 2, figsize=(12, 9))
# 自定义Nature风格的颜色
nature_blue = '#2C3E50' # 深蓝
nature_green = '#27AE60' # 温和绿
nature_red = '#C0392B' # 深红
nature_purple = '#8E44AD' # 紫罗兰

# 1. 原始信号 ys vs xs
axs[0, 0].plot(xs, ys, label='Voltage', color=nature_blue)
axs[0, 0].set_xlabel('Time(ns)')
axs[0, 0].set_ylabel('Voltage(V)')
axs[0, 0].grid(True)

# 2. bin_means（每个周期最后 integration_length 时间段的平均值）
# 2. bin_means 对应的电压步长，绘制时转换为 mV
axs[0, 1].plot(np.array(Voltage_steps) * 1000, 'o-', label='Bin Means', color=nature_green)
axs[0, 1].set_xlabel('Trigger count')
axs[0, 1].set_ylabel('Step (mV)')
axs[0, 1].grid(True)

# 3. DNL - 柱状图
axs[1, 0].bar(range(len(DNL)), DNL, color=nature_red, edgecolor='grey', alpha=0.7)
axs[1, 0].set_xlabel('Trigger count')
axs[1, 0].set_ylabel('DNL (LSB)')
axs[1, 0].axhline(0, color='black', linestyle='--', linewidth=1)
axs[1, 0].grid(True, axis='y', linestyle='--', alpha=0.6)

# 4. INL - 柱状图
axs[1, 1].bar(range(len(INL)), INL, color=nature_purple, edgecolor='grey', alpha=0.7)
axs[1, 1].set_xlabel('Trigger count')
axs[1, 1].set_ylabel('INL (LSB)')
axs[1, 1].axhline(0, color='black', linestyle='--', linewidth=1)
axs[1, 1].grid(True, axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
