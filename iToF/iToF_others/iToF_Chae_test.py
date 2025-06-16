import matplotlib.pyplot as plt
from iToF.iToF_others.iToF_Chae import  calculate_alphas
from iToF.Lib.BG_Counts import *

# 设置 start_time 列表（你可以根据需要调整）
start_time_list = np.linspace(10, 400, 142)  # 多个起始时间
num_runs = 100  # 每个 start_time 跑多少次

# 存储误差和标准差
mean_errors_t1 = []
std_devs_t1 = []
mean_errors_t2 = []
std_devs_t2 = []

for start_time in start_time_list:
    error_t1 = []
    error_t2 = []

    for _ in range(num_runs):
            # 生成事件
        pulse_events = generate_pulse_events(start_time, 10, 1000 + start_time)
        bg_events = generate_BG_events(0, 2000)
        total_events = pulse_events + bg_events
        events = apply_dead_time(total_events, 5)

        # 计算 t1 和 t2
        t1, t2 = calculate_alphas(events, 1000)

        if t1 is not None and t2 is not None:
            error_t1.append(abs(t1 - start_time))
            error_t2.append(abs(t2 - start_time))

    # 计算统计量
    mean_errors_t1.append(np.mean(error_t1))
    std_devs_t1.append(np.std(error_t1))
    mean_errors_t2.append(np.mean(error_t2))
    std_devs_t2.append(np.std(error_t2))

# 绘图准备
x = start_time_list

# 创建两个子图：一个显示平均误差，一个显示标准差
fig, axs = plt.subplots(2, 1, figsize=(9, 6))

# 平均误差
axs[0].plot(x, mean_errors_t1, 'o-', label='t1 Mean Error')
axs[0].plot(x, mean_errors_t2, 's-', label='t2 Mean Error')
axs[0].set_title('Mean Error vs Start Time')
axs[0].set_ylabel('Mean Error')
axs[0].grid(True)
axs[0].legend()

# 标准差
axs[1].plot(x, std_devs_t1, 'o-', label='t1 Std Dev', color='orange')
axs[1].plot(x, std_devs_t2, 's-', label='t2 Std Dev', color='green')
axs[1].set_title('Standard Deviation vs Start Time')
axs[1].set_xlabel('Start Time (ns)')
axs[1].set_ylabel('Std Dev')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()