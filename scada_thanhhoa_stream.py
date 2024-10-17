import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Cấu hình trang Streamlit
st.set_page_config(layout="wide")

# Tiêu đề của ứng dụng
st.title('Demo Dự đoán Bất thường cho Dữ liệu SCADA Thanh Hóa')

# Danh sách các thiết bị
list_deviced = [
    '66d92f94b7f41dd37df0634a',
    '66d92fabb7f41dd37df06351',
    '66d92fc2b7f41dd37df06355',
    '66d92fc6b7f41dd37df06359',
    '66d92fcbb7f41dd37df0635d',
    '66d92fcfb7f41dd37df06361'
]

# Danh sách các tham số (nếu có nhiều hơn 'P', hãy thêm vào đây)
list_pa = ['P']

# Tạo hai cột để chọn thiết bị và tham số
col1, col2 = st.columns(2)
with col1:
    select_deviced = st.selectbox('Chọn thiết bị', list_deviced)
with col2:
    select_parameter = st.selectbox('Chọn trạng thái', list_pa)

# Hàm để tải dữ liệu với caching để tăng tốc độ
@st.cache_data
def load_data(filename):
    chunk_size = 2000000  # Điều chỉnh giá trị này dựa trên khả năng của hệ thống
    chunks = []
    for chunk in pd.read_csv(filename, chunksize=chunk_size):
        chunks.append(chunk)
    data = pd.concat(chunks, axis=0)
    return data

# Đường dẫn tới file CSV (hãy chỉnh đường dẫn phù hợp với hệ thống của bạn)
csv_file_path = r'd:\2.WORK_DAILY\4.eKGIS_Data Science\Water\data\iot_water_thanhhoa.csv'
df = load_data(csv_file_path)

# Chuyển đổi timestamp thành datetime
df['datetime'] = pd.to_datetime(df['ts'], unit='s')

# Lọc các cột cần thiết
filter_df = df[['deviceid', 'datetime', 'dbl_v', 'parameter_key']]

# Lọc dữ liệu theo thiết bị và tham số đã chọn
df_device = df[df['deviceid'] == select_deviced].copy()
df_device = df_device[df_device['parameter_key'] == select_parameter]
df_device = df_device[['datetime', 'dbl_v']]

# Trích xuất giờ và phút, và tính tổng phút trong ngày
df_device['hour'] = df_device['datetime'].dt.hour
df_device['minute'] = df_device['datetime'].dt.minute
df_device["total_minute"] = df_device['minute'] + df_device['hour'] * 60

# Hàm để trực quan hóa dữ liệu bằng Matplotlib
def SensorViz(df, feature_X, feature_y, savefig=False):
    plt.figure(figsize=(15, 3))
    plt.scatter(df[feature_X], df[feature_y], color='blue', label='Giá trị thực tế')
    plt.plot(df[feature_X], df[feature_y], color='blue', linewidth=1)
    plt.title(f'{feature_X} vs {feature_y}', size=20)
    plt.xlabel(feature_X, size=15)
    plt.ylabel(feature_y, size=15)
    plt.grid(True)
    plt.legend()
    if savefig:
        plt.savefig(f"{feature_X}_vs_{feature_y}.png")
    st.pyplot(plt)

# Định nghĩa các đặc trưng và mục tiêu
feature_X = 'total_minute'
feature_y = 'dbl_v'

# Nhóm dữ liệu theo 'total_minute' và tính trung bình của 'dbl_v'
data_group = df_device.groupby(feature_X, as_index=False).mean()

# Trực quan hóa dữ liệu thực tế
SensorViz(data_group, feature_X, feature_y)

# Trích xuất X và y
X = data_group[[feature_X]].values
y = data_group[feature_y].values

# Định nghĩa thư mục lưu mô hình
model_directory = "saved_models"

# Định nghĩa tên tệp mô hình và bộ chuyển đổi dựa trên thiết bị
model_filename = f"linear_regression_model_{select_deviced}.pkl"
poly_filename = f"polynomial_features_{select_deviced}.pkl"

model_path = os.path.join(model_directory, model_filename)
poly_path = os.path.join(model_directory, poly_filename)


# Tải mô hình và bộ chuyển đổi
lin_reg_load = joblib.load(f"linear_regression_model{select_deviced}.pkl")
poly_reg_load = joblib.load(f"polynomial_features{select_deviced}.pkl")
# linear_regression_model66d92fcfb7f41dd37df06361

# Dự đoán y sử dụng mô hình
X_poly = poly_reg_load.transform(X)
y_predict = lin_reg_load.predict(X_poly)

# Tính toán residuals và độ lệch chuẩn
residuals = y - y_predict
std = np.std(residuals)

# Tính các biên giới trên và dưới
y_predict_upBound = y_predict + 3 * std
y_predict_lowBound = y_predict - 3 * std

results_df = data_group.copy()
results_df['y_predict'] = y_predict
results_df['y_predict_upBound'] = y_predict_upBound
results_df['y_predict_lowBound'] = y_predict_lowBound


# Biểu đồ So sánh Giá trị Thực tế và Dự đoán bằng Matplotlib
st.subheader('Biểu đồ So sánh Giá trị Thực tế và Dự đoán')

plt.figure(figsize=(18, 5))
plt.scatter(X, y, color="blue", label='Giá trị thực tế')
plt.plot(X, y_predict, color="red", linewidth=5, label='Giá trị dự đoán')
plt.plot(X, y_predict_upBound, color="green", linewidth=1, linestyle='--', label='Biên giới trên (y_predict + 3σ)')
plt.plot(X, y_predict_lowBound, color="green", linewidth=1, linestyle='--', label='Biên giới dưới (y_predict - 3σ)')

plt.title(f"{feature_X} vs {feature_y}", fontsize=18)
plt.xlabel(feature_X.capitalize(), fontsize=15)
plt.ylabel(feature_y.capitalize(), fontsize=15)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
st.pyplot(plt)

# Nhập dữ liệu mới để dự đoán
st.subheader('Dự đoán thử trên các giá trị mới')
new_minutes = st.text_input("Nhập các phút mới để dự đoán (cách nhau bằng dấu phẩy)", "10,20,30")

# Xử lý dữ liệu nhập vào và dự đoán
try:
    new_X = np.array([[int(x.strip())] for x in new_minutes.split(',')])
    new_X_poly = poly_reg_load.transform(new_X)
    new_y_pred = lin_reg_load.predict(new_X_poly)
except Exception as e:
    st.error("Vui lòng nhập các giá trị phút hợp lệ, cách nhau bằng dấu phẩy. Ví dụ: 10,20,30")

# Biểu đồ So sánh Giá trị Thực tế và Dự đoán bằng Plotly
st.subheader('Biểu đồ So sánh Giá trị Thực tế và Dự đoán (Plotly)')

fig = go.Figure()

# Đường giá trị thực tế
fig.add_trace(go.Scatter(
    x=results_df[feature_X],
    y=results_df[feature_y],
    mode='lines+markers',
    name='Giá trị thực tế',
    line=dict(color='blue')
))

# Đường dự đoán
fig.add_trace(go.Scatter(
    x=results_df[feature_X],
    y=results_df['y_predict'],
    mode='lines',
    name='Giá trị dự đoán',
    line=dict(color='red')
))

# Đường biên giới trên
fig.add_trace(go.Scatter(
    x=results_df[feature_X],
    y=results_df['y_predict_upBound'],
    mode='lines',
    name='Biên giới trên (y_predict + 3σ)',
    line=dict(color='green', dash='dash')
))

# Đường biên giới dưới
fig.add_trace(go.Scatter(
    x=results_df[feature_X],
    y=results_df['y_predict_lowBound'],
    mode='lines',
    name='Biên giới dưới (y_predict - 3σ)',
    line=dict(color='green', dash='dash'),
    fill='tonexty',  # Tạo vùng giữa hai biên giới
    fillcolor='rgba(0,255,0,0.1)'
))

# Thêm các điểm dự đoán thử nếu có
if 'new_X' in locals() and 'new_y_pred' in locals():
    fig.add_trace(go.Scatter(
        x=new_X.flatten(),
        y=new_y_pred,
        mode='markers',
        name='Dự đoán thử',
        marker=dict(color='orange', size=10, symbol='x')
    ))

    # Thêm chú thích cho các điểm dự đoán thử
    for i, (x_val, y_val) in enumerate(zip(new_X.flatten(), new_y_pred)):
        fig.add_annotation(
            x=x_val,
            y=y_val,
            text=f'({x_val}, {y_val:.2f})',
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
            font=dict(color='orange')
        )

# Cập nhật layout của biểu đồ
fig.update_layout(
    title='So Sánh Giữa Giá Trị Thực Tế và Dự Đoán với Biên Giới',
    xaxis_title=feature_X.capitalize(),
    yaxis_title=feature_y.capitalize(),
    legend=dict(x=0, y=1),
    template='plotly_white',
    width=1200,
    height=600
)

# Hiển thị biểu đồ Plotly
st.plotly_chart(fig, use_container_width=True)
