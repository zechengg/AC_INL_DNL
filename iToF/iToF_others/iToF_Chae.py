def calculate_alphas(times, total_time):
    """
    根据事件时间列表和总测量时间计算 alpha1 和 alpha2。

    参数：
        times       : 时间戳列表 (单位：ns)
        total_time  : 主测量时间长度 (单位：ns)，默认值为 400 ns

    返回：
        alpha1, alpha2, t1_calculated, t2_calculated
    """


    # 初始化四个计数器
    NP0 = 0  # [start_time, total_time]
    NP180 = 0  # [total_time, total_time + start_time]
    NP90 = 0  # [start_time, total_time / 2]
    NP270 = 0  # [total_time / 2, total_time + start_time]

    # 遍历所有事件时间
    for current_time in times:
        if 0 <= current_time <= total_time:
            NP0 += 1
            if current_time <= total_time / 2:
                NP90 += 1
            else:
                NP270 += 1
        elif total_time < current_time <= total_time + total_time / 2:
            NP180 += 1
            NP270 += 1  # 也属于后半段

    # 计算 alpha1
    try:
        alpha1 = NP180 / (NP0 + NP180)
    except ZeroDivisionError:
        alpha1 = float('nan')

    # 计算 alpha2
    try:
        numerator = NP90 - NP270
        denominator = abs(NP0 - NP180) + abs(NP90 - NP270)
        alpha2 = -0.5 * numerator / denominator
    except ZeroDivisionError:
        alpha2 = float('nan')

    # 新增计算
    t1_calculated = alpha1 * total_time
    t2_calculated = alpha2 * total_time

    return t1_calculated, t2_calculated