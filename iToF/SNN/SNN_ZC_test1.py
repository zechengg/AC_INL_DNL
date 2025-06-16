# predict.py
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from iToF.Lib.BG_Counts import *
from iToF.iToF_ZC.iToF_ZC import simulate_voltage_decay,period

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# 加载模型
model = load_model('quadratic_model.keras')

# 加载 scaler
scaler_x = joblib.load('scaler_x.pkl')  # 输入 y 的归一化器
scaler_y = joblib.load('scaler_y.pkl')  # 输出 x 的归一化器

# 定义预测函数
def predict_original_x(model, scaler_x, scaler_y, raw_y):
    """
    输入一个原始 y 值，返回预测的 x 值
    """
    raw_y_array = np.array([[raw_y]])
    y_scaled = scaler_x.transform(raw_y_array)
    x_scaled_pred = model.predict(y_scaled, verbose=0)
    x_original = scaler_y.inverse_transform(x_scaled_pred)
    return x_original[0][0]

def predict_original_x_batch(model, scaler_x, scaler_y, raw_y_list):
    """
    批量预测多个 y 值
    """
    raw_y_array = np.array(raw_y_list).reshape(-1, 1)
    y_scaled = scaler_x.transform(raw_y_array)
    x_scaled_pred = model.predict(y_scaled, verbose=0)
    x_original = scaler_y.inverse_transform(x_scaled_pred)
    return x_original.flatten().tolist()

def run_prediction_on_test_file(model, scaler_x, scaler_y, file_path=r'C:\Users\zeche\PycharmProjects\AC_INL_DNL\iToF\iToF_ZC\zc_SNN_Test_data.txt', output_file='results.csv'):
    """
    从文件中加载测试数据并进行预测，输出带真实值和预测值的 CSV 文件
    """
    # 读取测试文件
    data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    data.columns = ['x_true', 'y_input']  # 第一列是真实值，第二列是输入 y

    # 提取输入 y 并预测
    y_inputs = data['y_input'].values
    predicted_x_list = predict_original_x_batch(model, scaler_x, scaler_y, y_inputs)

    # 添加预测结果到 DataFrame
    data['x_pred'] = predicted_x_list

    # 重命名列方便阅读
    data.rename(columns={'x_true': 'True_x', 'y_input': 'Input_y', 'x_pred': 'Predicted_x'}, inplace=True)

    # 保存结果到文件
    data.to_csv(output_file, index=False)
    print(f"✅ 预测完成，结果已保存至 {output_file}")

    # 打印前几行查看结果
    print("\n📊 预测结果预览：")
    print(data.head())

    return data
# 示例使用
if __name__ == "__main__":
    # 单个预测

    # 调用函数进行预测并保存结果
    results_df = run_prediction_on_test_file(model, scaler_x, scaler_y, file_path=r'C:\Users\zeche\PycharmProjects\AC_INL_DNL\iToF\iToF_ZC\zc_SNN_Test_data.txt')

    grouped_stats = pd.DataFrame(columns=['True_x', 'count', 'mae', 'std'])

    # 获取所有唯一的真实值
    unique_true_values = results_df['True_x'].unique()

    # 遍历每个唯一的真实值
    for true_value in unique_true_values:
        # 提取该真实值对应的所有预测结果
        group = results_df[results_df['True_x'] == true_value]

        # 获取预测值
        predicted_values = group['Predicted_x']

        # 计算 MAE 和 STD
        mae = np.mean(np.abs(predicted_values - true_value))
        std = np.std(predicted_values)

        # 添加统计信息到新的 DataFrame
        new_row = {
            'True_x': true_value,
            'count': len(group),
            'mae': mae,
            'std': std
        }

        grouped_stats = pd.concat([grouped_stats, pd.DataFrame([new_row])], ignore_index=True)

    # 按 'True_x' 排序，让结果更清晰
    grouped_stats = grouped_stats.sort_values(by='True_x').reset_index(drop=True)

    # 打印前几行查看结果
    print("\n📊 按真实值分组的误差统计：")
    print(grouped_stats.head())
    mae = mean_absolute_error(results_df['True_x'], results_df['Predicted_x'])
    rmse = np.sqrt(mean_squared_error(results_df['True_x'], results_df['Predicted_x']))
    print(f"\n📊 MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    plt.figure(figsize=(12, 5))

    # MAE 曲线
    plt.subplot(1, 2, 1)
    plt.plot(grouped_stats['True_x'], grouped_stats['mae'], marker='o', linestyle='-')
    plt.title('MAE per True_x')
    plt.xlabel('True_x')
    plt.ylabel('MAE')
    plt.grid(True)

    # STD 曲线
    plt.subplot(1, 2, 2)
    plt.plot(grouped_stats['True_x'], grouped_stats['std'], marker='o', color='orange', linestyle='-')
    plt.title('Standard Deviation per True_x')
    plt.xlabel('True_x')
    plt.ylabel('STD')
    plt.grid(True)

    plt.tight_layout()
    plt.show()