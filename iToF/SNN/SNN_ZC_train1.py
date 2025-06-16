import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 读取数据
data = pd.read_csv(r'C:\Users\zeche\PycharmProjects\AC_INL_DNL\iToF\iToF_ZC\zc_SNN_Train_data.txt', delim_whitespace=True, header=None)

data.columns = ['x', 'y']  # 假设数据文件有两列

# 数据拆分
X = data['y'].values.reshape(-1, 1)  # 以y作为输入
y = data['x'].values  # 以x作为输出

# 数据归一化
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 创建模型
model = Sequential([
    Dense(64, input_dim=1, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# 保存模型
#model.save('quadratic_model.h5')
model.save('quadratic_model.keras')
# 保存 scaler
joblib.dump(scaler_x, 'scaler_x.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

# 如果需要逆归一化，使用以下代码
def inverse_transform(x_scaled):
    return scaler_y.inverse_transform(x_scaled)

# 使用模型预测并逆归一化
y_pred_scaled = model.predict(X_test)
y_pred = inverse_transform(y_pred_scaled)

# 打印一些预测结果用于验证
for i in range(5):
    print(f"真实值: {scaler_y.inverse_transform(y_test[i].reshape(-1, 1))[0][0]}, 预测值: {y_pred[i][0]}")