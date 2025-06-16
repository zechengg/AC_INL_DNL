def calculate_tof(distance, speed):
    """
    计算飞行时间（ToF）

    参数:
    distance (float): 目标与传感器之间的距离（单位: 米）
    speed (float): 信号传播速度（单位: 米/秒）

    返回:
    float: 飞行时间（单位: 秒）
    """
    if distance < 0 or speed <= 0:
        raise ValueError("距离必须 >= 0，速度必须 > 0")

    tof = distance / speed
    return tof


# 示例使用：
if __name__ == "__main__":
    # 已知条件
    distance = 5.0  # 距离：3 米
    speed_of_light = 299792458  # 光速（m/s）
    speed_of_sound = 343  # 声速（m/s），在空气中约 343 m/s

    # 使用光速计算 ToF
    tof_light = calculate_tof(distance, speed_of_light)
    print(f"使用光速计算，{distance} 米的 ToF 为: {tof_light * 1e9:.4f} 纳秒")

