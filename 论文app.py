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

# 3. 构建集成模型（XGB + ET + LGBM）
xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
et = ExtraTreesClassifier(n_estimators=100, random_state=42)
lgbm = LGBMClassifier(n_estimators=100, random_state=42)

model = VotingClassifier(estimators=[
    ('xgb', xgb),
    ('et', et),
    ('lgbm', lgbm)
], voting='hard')

model.fit(X_train, y_train_encoded)

# 4. 文件上传
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

    # 构造输入特征
    X_input = pd.DataFrame()
    for col_train, col_input in matching_columns.items():
        if col_input is not None:
            X_input[col_train] = input_data[col_input]
        else:
            X_input[col_train] = 0

    # 缺失值处理
    if X_input.isnull().values.any():
        st.warning("Some values are missing in the uploaded data. Missing values have been filled with 0.")
        X_input = X_input.fillna(0)

    # 5. 预测
    predicted_classes = model.predict(X_input)
    predicted_classes = label_encoder.inverse_transform(predicted_classes)

    # 使用其中一个模型计算置信度（e.g. LGBM）
    confidence_model = lgbm
    confidence_model.fit(X_train, y_train_encoded)
    predicted_probabilities = confidence_model.predict_proba(X_input)
    confidence_scores = np.max(predicted_probabilities, axis=1)

    # 添加预测与置信度
    input_data['Predicted Class'] = predicted_classes
    input_data['Confidence'] = confidence_scores

    # 显示预测结果
    st.write(input_data)

    # 结果下载按钮
    to_write = io.BytesIO()
    input_data.to_excel(to_write, index=False, engine='xlsxwriter')
    st.download_button("Download Prediction Results", data=to_write.getvalue(), file_name="predicted_results.xlsx")

    # 6. 绘制散点图（SiO2 vs MgO）
    img_path = r"MgO-SiO2.jpg"
    img = Image.open(img_path)

    if 'SiO2' in input_data.columns and 'MgO' in input_data.columns:
        sio2 = input_data['SiO2']
        mgo = input_data['MgO']
    else:
        st.error("The input Excel file is missing SiO2 or MgO columns.")
        sio2 = mgo = None

    if sio2 is not None and mgo is not None:
        plt.figure(figsize=(10, 10))
        min_sio2, max_sio2 = sio2.min() - 1, sio2.max() + 1
        min_mgo, max_mgo = mgo.min() - 1, mgo.max() + 1
        plt.imshow(img, extent=[min_sio2, max_sio2, min_mgo, max_mgo])

        unique_classes = label_encoder.classes_
        cmap = plt.get_cmap('tab10')
        class_colors = {class_name: cmap(i) for i, class_name in enumerate(unique_classes)}

        for class_name in unique_classes:
            class_indices = input_data['Predicted Class'] == class_name
            plt.scatter(sio2[class_indices], mgo[class_indices],
                        color=class_colors[class_name], label=class_name, alpha=0.6, s=300)

        plt.xlabel('SiO2', fontsize=16)
        plt.ylabel('MgO', fontsize=16)
        plt.title('Scatter Plot of SiO2 and MgO by Class', fontsize=18)
        plt.legend(ncol=5, loc='upper right')
        plt.tick_params(axis='both', which='both', length=0)
        st.pyplot(plt)

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

# 模型加载完成信息
st.success("The ensemble model (XGB + ET + LGBM) has been loaded and trained successfully.")
# 检查预测结果中是否包含 FAB、boninite、HMA 三种岩石类型
detected_classes = set(input_data['Predicted Class'].unique())
target_classes = {'FAB', 'boninite', 'HMA'}
intersection = detected_classes & target_classes

# 如果检测到目标岩石中的任意一种
if intersection:
    # 询问用户是否要继续进行初始俯冲事件判断
    if st.radio("Samples consistent with the initial subduction rock sequence were detected. Do you want to perform a follow-up judgment?", ["YES", "NO"]) == "YES":

        # 情况①：三种岩石都出现
        if intersection == target_classes:
            st.success("Three types of rocks (FAB, boninite, and HMA) were detected. Please enter their geological information.")

            ages = {}
            lons = {}
            lats = {}

            # 用户输入三种岩石的年龄、经度和纬度
            for rock in sorted(target_classes):
                ages[rock] = st.number_input(f"Enter the age of {rock} (Ma)", min_value=0.0, value=0.0, format="%.2f", key=f"{rock}_age")
                lons[rock] = st.number_input(f"Enter the longitude of {rock} (°)", format="%.2f", value=0.0, key=f"{rock}_lon")
                lats[rock] = st.number_input(f"Enter the latitude of {rock} (°)", format="%.2f", value=0.0, key=f"{rock}_lat")

            # 计算年龄和经纬度的范围
            age_range = max(ages.values()) - min(ages.values())
            lon_range = max(lons.values()) - min(lons.values())
            lat_range = max(lats.values()) - min(lats.values())

            # 判断是否满足时间与空间约束
            if age_range <= 10 and lon_range <= 5 and lat_range <= 5:
                st.success("According to the IBM regional rock sequence study, your study area may have an initial subduction event!")
            else:
                if age_range > 10:
                    st.warning("According to the IBM regional rock sequence study, your study area may have an initial subduction event, but the age range of the samples varies greatly. Please consider the actual situation in your study area.")
                if lon_range > 5 or lat_range > 5:
                    st.warning("According to the rock stratigraphic study of the IBM region, there may be an initial subduction event in your study area, but the longitude and latitude ranges of the samples vary greatly. Please consider the actual situation in your study area.")

        # 情况②：只出现两种岩石类型
        elif len(intersection) == 2:
            missing = list(target_classes - intersection)[0]  # 找出缺失的岩石类型
            st.info(f"Two rock types were detected: {', '.join(sorted(intersection))}. The missing sample is: {missing}.")

            ages = {}
            lons = {}
            lats = {}

            # 用户输入两种岩石的地质信息
            for rock in sorted(intersection):
                ages[rock] = st.number_input(f"Enter the age of {rock} (Ma)", min_value=0.0, value=0.0, format="%.2f", key=f"{rock}_age")
                lons[rock] = st.number_input(f"Enter the longitude of {rock} (°)", format="%.2f", value=0.0, key=f"{rock}_lon")
                lats[rock] = st.number_input(f"Enter the latitude of {rock} (°)", format="%.2f", value=0.0, key=f"{rock}_lat")

            age_range = max(ages.values()) - min(ages.values())
            lon_range = max(lons.values()) - min(lons.values())
            lat_range = max(lats.values()) - min(lats.values())

            # 判断年龄与经纬度条件
            if age_range <= 10 and lon_range <= 5 and lat_range <= 5:
                st.success(f"According to the IBM regional rock sequence study, your study area may have an initial subduction event! However, one type of sample ({missing}) is missing. Please add it.")
            else:
                if age_range > 10:
                    st.warning("According to the IBM regional rock sequence study, your study area may have an initial subduction event, but the age range of the samples varies greatly. Please consider the actual situation in your study area.")
                if lon_range > 5 or lat_range > 5:
                    st.warning("According to the rock stratigraphic study of the IBM region, there may be an initial subduction event in your study area, but the longitude and latitude ranges of the samples vary greatly. Please consider the actual situation in your study area.")

        # 情况③：只出现一种岩石类型
        elif len(intersection) == 1:
            missing = list(target_classes - intersection)
            st.info(f"Only one rock type was detected: {list(intersection)[0]}. Two samples are missing: {missing[0]} and {missing[1]}. Please add them.")
