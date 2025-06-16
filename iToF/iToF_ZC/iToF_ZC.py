import numpy as np
import matplotlib.pyplot as plt
#from iToF.iToF_ZC.iToF_ZC import period, total_time, simulate_voltage_decay
from iToF.Lib.BG_Counts import *
# 参数设置
initial_voltage = 1800     # 初始电压 (mV)
period = 8               # 放电周期 (ns)
total_time = 500           # 总测量时间 (ns)
initial_deltaV = 30
final_deltaV = 0
num_runs = 50
# 电压下降幅值随时间线性变化：t=0 时为20mV，t=400ns 时为10mV


def get_drop_amount(t):
    slope = (final_deltaV - initial_deltaV) / total_time  # 斜率
    return initial_deltaV + slope * t


def generate_Vdate_train():
    # 初始化 start_time 列表：从 10 ns 到 400 ns，共10个点
    start_times = np.random.uniform(low=20, high=500, size=100000)

    # 存储每个 start_time 的最终电压
    final_voltages = []

    # 对每个 start_time 进行一次模拟
    for start_time in start_times:
        voltage = initial_voltage
        current_time = start_time

        while current_time <= total_time:
            drop = get_drop_amount(current_time)
            voltage -= drop
            jitter = np.random.normal(loc=0.0, scale=0.2)
            # 更新下一次放电时间
            next_time = current_time + period + jitter
            current_time = next_time

        final_voltages.append(voltage)
        print(f"start_time = {start_time:.2f} ns → 最终电压: {voltage:.2f} mV")

    # 指定要保存的文件名
    filename = "zc_SNN_Train_data.txt"

    # 写入文件
    with open(filename, 'w') as f:
        for xi, yi in zip(start_times, final_voltages):
            f.write(f"{xi:.4f} {yi:.4f}\n")  # 保留4位小数，也可以根据需要修改格式

    # 绘图展示
    plt.figure(figsize=(10, 6))
    plt.plot(start_times, final_voltages, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Start Time (ns)', fontsize=12)
    plt.ylabel('Final Voltage (mV)', fontsize=12)
    plt.title('Final Voltage vs Start Time', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def generate_Vdata_true():
    # 初始化 start_time 列表：从 10 ns 到 400 ns，共10个点
    start_times = np.linspace(20, 500, 25003)

    # 存储每个 start_time 的最终电压
    final_voltages = []

    # 对每个 start_time 进行一次模拟
    for start_time in start_times:
        voltage = initial_voltage
        current_time = start_time

        while current_time <= total_time:
            drop = get_drop_amount(current_time)
            voltage -= drop

            # 更新下一次放电时间
            next_time = current_time + period #+ jitter_ns
            current_time = next_time

        final_voltages.append(voltage)

    # 指定要保存的文件名
    filename = "zc_time_voltage_data.txt"
    print(filename)
    # 写入文件
    with open(filename, 'w') as f:
        for xi, yi in zip(start_times, final_voltages):
            f.write(f"{xi:.4f} {yi:.4f}\n")  # 保留4位小数，也可以根据需要修改格式
    # 绘图展示
    plt.figure(figsize=(10, 6))
    plt.plot(start_times, final_voltages, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Start Time (ns)', fontsize=12)
    plt.ylabel('Final Voltage (mV)', fontsize=12)
    plt.title('Final Voltage vs Start Time', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def generate_Vdate_test():

    # 初始化 start_time 列表：从 10 ns 到 400 ns，共10个点
    start_times = np.linspace(20, 400, 52)

    # 存储每个 start_time 的最终电压
    real_times = []
    final_voltages = []
    for start_time in start_times:

        for _ in range(num_runs):
            # 生成事件
            events = generate_pulse_events(start_time, period, total_time, 0.2)
            bg_events = generate_BG_events(0, total_time)
            final_events = apply_dead_time(events + bg_events, 5)
            # 计算 t1 和 t2
            voltage = simulate_voltage_decay(final_events)
            real_times.append(start_time)
            final_voltages.append(voltage)

    # 指定要保存的文件名
    filename = "zc_SNN_Test_data.txt"

    # 写入文件
    with open(filename, 'w') as f:
        for xi, yi in zip(real_times, final_voltages):
            f.write(f"{xi:.4f} {yi:.4f}\n")  # 保留4位小数，也可以根据需要修改格式

def simulate_voltage_decay(times):

    """
    模拟不同起始时间下的电压衰减过程，并绘制最终电压与起始时间的关系图。

    返回：
        start_times: 所有起始时间列表
        final_voltages: 对应的最终电压列表
    """
    voltage = initial_voltage
    # 对每个 start_time 进行一次模拟
    for time in times:
        if time <= total_time:
            drop = get_drop_amount(time)
            voltage -= drop

    return voltage

if __name__ == '__main__':
    generate_Vdate_train()
    #generate_Vdate_test()
    #generate_Vdata_true()






