import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from PIL import Image
import io

# 页面配置
st.set_page_config(page_title="Rock Classification Prediction", layout="wide")

# 1. 加载训练集
train_file_path = r"FAB-Boninite-HMA-IAT-CA.xlsx"
train_data = pd.read_excel(train_file_path)

# 2. 数据预处理
X_train = train_data.drop(train_data.columns[0], axis=1)
y_train = train_data.iloc[:, 0]

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# 3. 构建集成模型
xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
et = ExtraTreesClassifier(n_estimators=100, random_state=42)
lgbm = LGBMClassifier(n_estimators=100, random_state=42)

model = VotingClassifier(estimators=[
    ('xgb', xgb),
    ('et', et),
    ('lgbm', lgbm)
], voting='hard')

model.fit(X_train, y_train_encoded)

# 4. 上传文件
uploaded_file = st.file_uploader("Upload a new Excel file for prediction", type=["xlsx"])
if uploaded_file is not None:
    input_data = pd.read_excel(uploaded_file)

    # 匹配列名
    matching_columns = {}
    for col_train in X_train.columns:
        for col_input in input_data.columns:
            if col_input.lower().startswith(col_train.lower()):
                matching_columns[col_train] = col_input
                break
        if col_train not in matching_columns:
            matching_columns[col_train] = None

    X_input = pd.DataFrame()
    for col_train, col_input in matching_columns.items():
        if col_input is not None:
            X_input[col_train] = input_data[col_input]
        else:
            X_input[col_train] = 0

    # 缺失值填充
    if X_input.isnull().values.any():
        st.warning("Some values are missing in the uploaded data. Missing values have been filled with 0.")
        X_input = X_input.fillna(0)

    # 5. 模型预测
    predicted_classes = model.predict(X_input)
    predicted_classes = label_encoder.inverse_transform(predicted_classes)

    confidence_model = lgbm
    confidence_model.fit(X_train, y_train_encoded)
    predicted_probabilities = confidence_model.predict_proba(X_input)
    confidence_scores = np.max(predicted_probabilities, axis=1)

    input_data['Predicted Class'] = predicted_classes
    input_data['Confidence'] = confidence_scores

    # 显示结果
    st.write(input_data)

    # 下载按钮
    to_write = io.BytesIO()
    input_data.to_excel(to_write, index=False, engine='xlsxwriter')
    st.download_button("Download Prediction Results", data=to_write.getvalue(), file_name="predicted_results.xlsx")

# 6. 绘制 SiO2-MgO 散点图（带背景图）

img_path = r"MgO-SiO2.jpg"  # 请确保该路径下图片文件存在

try:
    img = Image.open(img_path)

    # 检查必要的列
    if 'SiO2' not in input_data.columns or 'MgO' not in input_data.columns:
        st.error("输入的 Excel 文件缺少 'SiO2' 或 'MgO' 列。")
    else:
        sio2 = input_data['SiO2']
        mgo = input_data['MgO']

        plt.figure(figsize=(10, 10))

        # 设置背景图范围（根据背景图标注范围调整）
        plt.imshow(img, extent=[45, 70, 0, 25], aspect='auto', zorder=0)

        # 获取预测类别及其颜色
        unique_classes = np.unique(predicted_classes)
        cmap = plt.get_cmap('tab10')
        class_colors = {class_name: cmap(i) for i, class_name in enumerate(unique_classes)}

        # 绘制每类点
        for class_name in unique_classes:
            class_indices = predicted_classes == class_name
            plt.scatter(sio2[class_indices], mgo[class_indices],
                        color=class_colors[class_name], label=class_name,
                        alpha=0.8, s=150, edgecolor='black', linewidth=0.5, zorder=1)

        plt.xlim(45, 70)
        plt.ylim(0, 25)
        plt.xlabel('SiO₂', fontsize=14)
        plt.ylabel('MgO', fontsize=14)
        plt.title('SiO₂-MgO 分类图（带背景图）', fontsize=16)
        plt.legend(title="预测类型", loc='upper right', fontsize=12)
        plt.grid(False)

        # 可选隐藏坐标轴刻度
        plt.xticks([])
        plt.yticks([])

        st.pyplot(plt)

except FileNotFoundError:
    st.error("未找到背景图 'MgO-SiO2.jpg'，请检查文件路径是否正确。")
except Exception as e:
    st.error(f"绘图失败：{e}")

    # 7. 置信度分布图
    for class_name in unique_classes:
        class_confidences = confidence_scores[input_data['Predicted Class'] == class_name]
        if len(class_confidences) == 0:
            continue

        plt.figure(figsize=(10, 6))
        plt.hist(class_confidences, bins=20, density=True, alpha=0.7,
                 color=class_colors[class_name], label=f'Predicted Class: {class_name}')

        density = gaussian_kde(class_confidences)
        xs = np.linspace(min(class_confidences), max(class_confidences), 200)
        plt.plot(xs, density(xs), 'r-', label='Fitting Curve')

        plt.xlabel('Confidence', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title(f'Confidence Distribution for Class {class_name}', fontsize=16)
        plt.legend()
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        st.image(buf, caption=f'Confidence Distribution for {class_name}')
        plt.close()

    # 8. 初始俯冲事件判断
    detected_classes = set(input_data['Predicted Class'].unique())
    target_classes = {'FAB', 'boninite', 'HMA'}
    intersection = detected_classes & target_classes

    if intersection:
        if st.radio("Samples consistent with the initial subduction rock sequence were detected. Do you want to perform a follow-up judgment?", ["YES", "NO"]) == "YES":
            if intersection == target_classes:
                st.success("Three types of rocks (FAB, boninite, and HMA) were detected. Please enter their geological information.")

                ages, lons, lats = {}, {}, {}
                for rock in sorted(target_classes):
                    ages[rock] = st.number_input(f"Enter the age of {rock} (Ma)", min_value=0.0, value=0.0, format="%.2f", key=f"{rock}_age")
                    lons[rock] = st.number_input(f"Enter the longitude of {rock} (°)", format="%.2f", value=0.0, key=f"{rock}_lon")
                    lats[rock] = st.number_input(f"Enter the latitude of {rock} (°)", format="%.2f", value=0.0, key=f"{rock}_lat")

                age_range = max(ages.values()) - min(ages.values())
                lon_range = max(lons.values()) - min(lons.values())
                lat_range = max(lats.values()) - min(lats.values())

                if age_range <= 10 and lon_range <= 5 and lat_range <= 5:
                    st.success("According to the IBM regional rock sequence study, your study area may have an initial subduction event!")
                else:
                    if age_range > 10:
                        st.warning("Age range too large. Check the geological context.")
                    if lon_range > 5 or lat_range > 5:
                        st.warning("Spatial distribution too wide. Check the geological context.")

            elif len(intersection) == 2:
                missing = list(target_classes - intersection)[0]
                st.info(f"Two rock types were detected: {', '.join(sorted(intersection))}. The missing sample is: {missing}.")

                ages, lons, lats = {}, {}, {}
                for rock in sorted(intersection):
                    ages[rock] = st.number_input(f"Enter the age of {rock} (Ma)", min_value=0.0, value=0.0, format="%.2f", key=f"{rock}_age")
                    lons[rock] = st.number_input(f"Enter the longitude of {rock} (°)", format="%.2f", value=0.0, key=f"{rock}_lon")
                    lats[rock] = st.number_input(f"Enter the latitude of {rock} (°)", format="%.2f", value=0.0, key=f"{rock}_lat")

                age_range = max(ages.values()) - min(ages.values())
                lon_range = max(lons.values()) - min(lons.values())
                lat_range = max(lats.values()) - min(lats.values())

                if age_range <= 10 and lon_range <= 5 and lat_range <= 5:
                    st.success(f"Possible subduction event! But {missing} sample is missing.")
                else:
                    if age_range > 10:
                        st.warning("Age range too large.")
                    if lon_range > 5 or lat_range > 5:
                        st.warning("Spatial range too wide.")

            elif len(intersection) == 1:
                missing = list(target_classes - intersection)
                st.info(f"Only one rock type was detected: {list(intersection)[0]}. Two samples are missing: {missing[0]} and {missing[1]}. Please add them.")

# 模型加载完成提示
st.success("The ensemble model (XGB + ET + LGBM) has been loaded and trained successfully.")
