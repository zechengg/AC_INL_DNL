import numpy as np

def generate_pulse_events(start_time, period, total_time, jitter_std = 0.2):
    """
    生成周期性脉冲事件的时间列表。

    参数：
        start_time   : 起始时间 (单位：ns)
        period       : 周期间隔 (单位：ns)
        total_time   : 最大时间上限 (单位：ns)

    返回：
        times        : 包含所有生成时间点的列表
    """
    times = []
    current_time = start_time

    while current_time <= total_time:
        jitter = np.random.normal(loc=0.0, scale=jitter_std)
        times.append(current_time + jitter)
        current_time += period

    return times



def generate_BG_events(count_rate_MHz, window_duration_ns):
    """
    模拟 SPAD 在环境光下的输出事件（触发时间点）

    参数：
        count_rate_MHz      : 环境光引起的平均计数率 (MHz)
        window_duration_ns  : 时间窗口长度 (ns)

    返回：
        times_with_jitter   : 包含抖动的事件时间点列表（单位：ns）
    """
    if count_rate_MHz == 0:
        return []
    # 内部定义抖动标准差（ns），不再作为输入参数
    jitter_sigma_ns = 0.1

    # 平均事件间隔（ns）
    mean_interval_ns = 1e3 / count_rate_MHz

    # 泊松过程生成事件时间点
    times = []
    current_time = 0.0
    while current_time < window_duration_ns:
        interval = np.random.exponential(mean_interval_ns)
        current_time += interval
        if current_time < window_duration_ns:
            times.append(current_time)
    return times

def apply_dead_time(times, dead_time):
    """
    对输入的时间列表应用死时间（dead time）过滤规则。

    参数：
        times      : 已排序的时间点列表 (单位：ns)
        dead_time  : 死时间阈值 (单位：ns)

    返回：
        filtered_times : 应用死时间规则后的新时间点列表
    """

    # 确保输入列表已排序
    times = sorted(times)

    filtered_times = [times[0]]  # 保留第一个事件

    for t in times[1:]:
        if t - filtered_times[-1] >= dead_time:
            filtered_times.append(t)

    return filtered_times





