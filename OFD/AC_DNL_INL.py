import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 手动修改这个值来控制保留多少个点
num_points_to_keep = 65
period = 10
time_start = 2
time = num_points_to_keep * period
# ====== 数据处理函数 ====== #
def find_most_common_y_with_real_x_center(data_file, decimal_places=2):
    interval_y_x_map = defaultdict(lambda: defaultdict(list))

    with open(data_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            x_str, y_str = line.strip().split()
            x = float(x_str)
            y = float(y_str)

            x = x * 1e9  # 步骤一：x乘以1e9

            if x < 12:
                continue

            interval_start = ((int(x) - time_start) // period) * period + time_start
            interval = (interval_start, interval_start + period-2)

            y_rounded = round(y, decimal_places)

            interval_y_x_map[interval][y_rounded].append(x)

    results = []

    for interval, y_x_list in interval_y_x_map.items():
        interval_start, interval_end = interval

        # 收集该区间下所有 (x, y)，并排序
        all_points = []
        for y_rounded, x_list in y_x_list.items():
            for x in x_list:
                all_points.append((x, y_rounded))

        # 按照 x 排序
        all_points.sort()

        # 如果没有数据点，跳过
        if not all_points:
            continue

        total_area = 0.0
        total_length = 0.0

        # 积分累加（梯形法）
        for i in range(1, len(all_points)):
            x_prev, y_prev = all_points[i - 1]
            x_curr, y_curr = all_points[i]

            dx = x_curr - x_prev
            avg_y = (y_prev + y_curr) / 2
            area = dx * avg_y

            total_area += area

        # 总区间长度
        total_length = interval_end - interval_start

        # 计算平均 y 值（积分均值）
        average_y = total_area / total_length if total_length > 0 else float('nan')

        # 结果记录
        results.append(((interval_start + interval_end) / 2, average_y))

    results.sort()

    # 只保留前num_points_to_keep个点
    result_subset = results[:num_points_to_keep]

    return result_subset


# ====== 新增功能：计算 DNL 和 INL，并转为 LSB 单位 ====== #
def calculate_dnl_inl_in_lsb(results_subset):
    ys = [y for x, y in results_subset]
    n = len(ys)

    if n < 2:
        raise ValueError("至少需要2个点来计算 DNL/INL")

    ideal_step = (ys[-1] - ys[0]) / (n - 1)  # 基于 subset 计算

    dnl_full = []
    voltage_steps = []  # 👈 新增：用于保存 actual_step

    for i in range(n - 1):
        actual_step = ys[i + 1] - ys[i]
        voltage_steps.append(actual_step)  # 👈 保存 actual_step 到数组中
        dnl_full.append((actual_step - ideal_step) / ideal_step)  # 转换为 LSB

    inl = []
    for i in range(n):
        ideal_y = ys[0] + i * ideal_step
        inl_value = (ys[i] - ideal_y) / ideal_step
        inl.append(inl_value)

    return dnl_full, inl, ideal_step, voltage_steps  # 👈 返回新增的 steps 数组


# ====== 主程序入口 ====== #
if __name__ == '__main__':
    result_subset = find_most_common_y_with_real_x_center('../data.txt', decimal_places=5)

    print("✅ 最终用于分析的点数:", len(result_subset))
    if len(result_subset) < num_points_to_keep:
        print(f"⚠️ 注意：原始数据不足{num_points_to_keep}个点，仅提取到 {len(result_subset)} 个点。")
    else:
        print(f"✅ 数据充足，使用前{num_points_to_keep}个点进行分析。")

    print(f"\n处理后的结果（前{num_points_to_keep}个点）：")
    for x, y in result_subset:
        print(f"x_center={x:.2f}, y={y:.5f}")

    dnl_lsb, inl_lsb, ideal_step, v_steps = calculate_dnl_inl_in_lsb(result_subset)

    print("\n✅ 实际电压步长 voltage_steps:")
    for i, step in enumerate(v_steps):
        print(f"Step {i}: {step:.6f} V")

    print(f"\n使用的 LSB 值（理想步长）: {ideal_step:.6f}")
    print("\nDNL（差分非线性，单位 LSB）：")
    for i, d in enumerate(dnl_lsb):
        print(f"点{i} -> 点{i+1}: DNL = {d:.4f} LSB")

    print("\nINL（积分非线性，单位 LSB）：")
    for i, v in enumerate(inl_lsb):
        print(f"点{i}: INL = {v:.4f} LSB")

    # ====== 读取原始数据并过滤 x < 650 的点 ====== #
    raw_xs = []
    raw_ys = []
    with open('../data.txt', 'r') as f:
        for line in f:
            if not line.strip():
                continue
            x_str, y_str = line.strip().split()
            x = float(x_str) * 1e9
            y = float(y_str)
            if x < num_points_to_keep * period:
                raw_xs.append(x)
                raw_ys.append(y)

    # ====== 设置全局字体大小 ====== #
    plt.rcParams.update({
        'font.size': 18,  # 默认字体大小
        'axes.titlesize': 18,  # 子图标题字体大小
        'axes.labelsize': 18,  # 坐标轴标签字体大小
        'xtick.labelsize': 18,  # x轴刻度字体大小
        'ytick.labelsize': 18,  # y轴刻度字体大小
        'legend.fontsize': 18,  # 图例字体大小
        'figure.titlesize': 18  # 整个 figure 的标题字体大小（可选）
    })
    xs = [x for x, y in result_subset]
    ys = [y for x, y in result_subset]
    indices = list(range(len(xs)))
    inl_values = inl_lsb
    dnl_values = dnl_lsb
    step_values = v_steps

    # 🟡 使用 2x2 的网格布局
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    # ========== 子图1：原始数据点（放在左上）==========
    ax[0, 0].scatter(raw_xs, raw_ys, s=5, color='gray', alpha=0.7)

    ax[0, 0].set_xlabel('Time (ns)')
    ax[0, 0].set_ylabel('Vout (V)')
    #ax[0, 0].set_xlim(0, time+period)
    ax[0, 0].grid(True)
    #ax[0, 0].legend()

    # ========== 子图2：电压步长（放在右上）==========
    ax[0, 1].scatter(range(len(step_values)), step_values, color='green', alpha=0.7, label='Voltage Steps', zorder=2)

    # 画一条水平的理想步长线（ideal_step），zorder 确保它在点的下方
    ax[0, 1].axhline(ideal_step, color='red', linestyle='--', linewidth=1.2, label=f'Average Step({ideal_step:.4f} V)',
                     zorder=1)

    # 设置标签和样式
    ax[0, 1].set_xlabel('Step Index')
    ax[0, 1].set_ylabel('Voltage Step (V)')
    ax[0, 1].grid(True)
    ax[0, 1].legend()

    # ========== 子图3：INL（放在左下）==========
    ax[1, 0].bar(indices, inl_values, color='green', alpha=0.7, label='INL (LSB)')
    ax[1, 0].axhline(0, color='black', linewidth=0.8)

    ax[1, 0].set_xlabel('Step Index')
    ax[1, 0].set_ylabel('INL (LSB)')
    ax[1, 0].grid(True)
    ax[1, 0].legend()

    # ========== 子图4：DNL（放在右下）==========
    ax[1, 1].bar(indices[1:], dnl_values, color='orange', alpha=0.7, label='DNL (LSB)')
    ax[1, 1].axhline(0, color='black', linewidth=0.8)
    # 🟢 设置 y 轴刻度间隔为 0.2，确保 y=0 上下都有清晰的网格线

    ax[1, 1].set_ylim(bottom=min(dnl_values) * 1.1, top=max(max(dnl_values) * 1.1, 0.05))
    ax[1, 1].set_xlabel('Step Index ')
    ax[1, 1].set_ylabel('DNL (LSB)')
    ax[1, 1].grid(True)
    ax[1, 1].legend()

    plt.tight_layout()
    plt.show()