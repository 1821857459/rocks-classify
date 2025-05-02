import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
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

# 构建集成模型（XGB + ET + 优化版 LGBM）
xgb = XGBClassifier(n_estimators=100, eval_metric='mlogloss', random_state=42)
et = ExtraTreesClassifier(n_estimators=100, random_state=42)
lgbm = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42
)

# 使用硬投票来计算最终预测结果
hard_voting_model = VotingClassifier(estimators=[
    ('xgb', xgb),
    ('et', et),
    ('lgbm', lgbm)
], voting='hard')

hard_voting_model.fit(X_train, y_train_encoded)

# 使用软投票来计算预测可信度（概率）
soft_voting_model = VotingClassifier(estimators=[
    ('xgb', xgb),
    ('et', et),
    ('lgbm', lgbm)
], voting='soft')

soft_voting_model.fit(X_train, y_train_encoded)

# 4. 文件上传
uploaded_file = st.file_uploader("Upload a new Excel file for prediction", type=["xlsx"])
if uploaded_file is not None:
    try:
        input_data = pd.read_excel(uploaded_file)
        st.write("File uploaded successfully!")
        st.write(input_data.head())  # Print the first few rows of the uploaded data for inspection

        # 数据预处理
        # 匹配列名（忽略大小写和多余的后缀）
        matching_columns = {}
        processed_train_columns = [col.lower().strip() for col in X_train.columns]
        processed_input_columns = [col.lower().strip() for col in input_data.columns]

        for col_train, processed_col_train in zip(X_train.columns, processed_train_columns):
            matched = False
            for col_input, processed_col_input in zip(input_data.columns, processed_input_columns):
                if processed_col_input.startswith(processed_col_train):
                    matching_columns[col_train] = col_input
                    matched = True
                    break
            if not matched:
                matching_columns[col_train] = None

        # Prepare input data
        X_input = pd.DataFrame()
        for col_train, col_input in matching_columns.items():
            if col_input is not None:
                X_input[col_train] = input_data[col_input]
            else:
                X_input[col_train] = 0  # Fill missing columns with 0

        # 使用硬投票模型进行预测（硬投票选择类别）
        predicted_classes_hard = hard_voting_model.predict(X_input)
        predicted_classes_hard = label_encoder.inverse_transform(predicted_classes_hard)

        # 使用软投票模型计算每个样本的预测概率（软投票为可信度计算提供支持）
        predicted_probs_soft = soft_voting_model.predict_proba(X_input)

        # 获取每个预测类别的最大概率作为可信度
        confidence_scores = np.max(predicted_probs_soft, axis=1)

        # 将预测结果和可信度加入输入数据
        input_data['Predicted Class (Hard Voting)'] = predicted_classes_hard
        input_data['Confidence (Soft Voting)'] = confidence_scores

        # 显示预测结果
        st.write(input_data)

        # 结果下载按钮
        to_write = io.BytesIO()
        input_data.to_excel(to_write, index=False, engine='xlsxwriter')
        st.download_button("Download Prediction Results", data=to_write.getvalue(), file_name="predicted_results.xlsx")

        # 6. 绘制 SiO2-MgO 散点图
        img_path = r"MgO-SiO2.jpg"  # <-- Replace with your local image path
        try:
            img = Image.open(img_path)
            if 'SiO2' in input_data.columns and 'MgO' in input_data.columns:
                sio2 = input_data['SiO2']
                mgo = input_data['MgO']
            else:
                st.error("The input Excel file is missing SiO2 or MgO columns.")

            # Adjust the extent parameters based on the actual coordinates of the image
            plt.figure(figsize=(10, 10))
            plt.imshow(img, extent=[45, 70, 0, 25], aspect='auto', zorder=0)

            unique_classes = np.unique(predicted_classes_hard)
            cmap = plt.get_cmap('tab10')
            class_colors = {class_name: cmap(i) for i, class_name in enumerate(unique_classes)}

            # Plot each class's points within the defined coordinate range
            for class_name in unique_classes:
                class_indices = predicted_classes_hard == class_name
                plt.scatter(sio2[class_indices], mgo[class_indices],
                            color=class_colors[class_name], label=class_name, alpha=0.7, s=150, edgecolor='k', zorder=1)

            # Set plot settings
            plt.xlim(45, 70)  # Limit the x-axis to match the extent of the image
            plt.ylim(0, 25)   # Limit the y-axis to match the extent of the image
            plt.xlabel('SiO2', fontsize=16)
            plt.ylabel('MgO', fontsize=16)
            plt.title('Scatter Plot of SiO2 and MgO by Class', fontsize=18)
            plt.legend(ncol=5, loc='upper right')
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            st.pyplot(plt)

        except Exception as e:
            st.error(f"Failed to load or plot image: {e}")

        # 检查是否检测到 FAB、boninite、HMA 岩石类型
        detected_classes = set(input_data['Predicted Class (Hard Voting)'].unique())
        target_classes = {'FAB', 'boninite', 'HMA'}
        intersection = detected_classes & target_classes

        # 如果检测到目标岩石类型
        if intersection:
            # 询问用户是否进行后续的俯冲事件判断
            if st.radio("Samples consistent with the initial subduction rock sequence were detected. Do you want to perform a follow-up judgment?", ["YES", "NO"]) == "YES":

                # 情况 1：三种岩石类型都出现
                if intersection == target_classes:
                    st.success("Three types of rocks (FAB, boninite, and HMA) were detected. Please enter their geological information.")

                    ages = {}
                    lons = {}
                    lats = {}

                    # 用户输入每种岩石类型的年龄、经度、纬度信息
                    for rock in sorted(target_classes):
                        ages[rock] = st.number_input(f"Enter the age of {rock} (Ma)", min_value=0.0, value=0.0, format="%.2f", key=f"{rock}_age")
                        lons[rock] = st.number_input(f"Enter the longitude of {rock} (°)", format="%.2f", value=0.0, key=f"{rock}_lon")
                        lats[rock] = st.number_input(f"Enter the latitude of {rock} (°)", format="%.2f", value=0.0, key=f"{rock}_lat")

                    # 计算年龄、经纬度范围
                    age_range = max(ages.values()) - min(ages.values())
                    lon_range = max(lons.values()) - min(lons.values())
                    lat_range = max(lats.values()) - min(lats.values())

                    # 检查时间和空间条件是否符合
                    if age_range <= 10 and lon_range <= 5 and lat_range <= 5:
                        st.success("According to the IBM regional rock sequence study, your study area may have an initial subduction event!")
                    else:
                        if age_range > 10:
                            st.warning("The age range of the samples is too large. Please consider the actual situation in your study area.")
                        if lon_range > 5 or lat_range > 5:
                            st.warning("The longitude and latitude ranges of the samples vary greatly. Please consider the actual situation in your study area.")

                # 情况 2：只检测到两种岩石类型
                elif len(intersection) == 2:
                    missing = list(target_classes - intersection)[0]  # 找出缺失的岩石类型
                    st.info(f"Two rock types were detected: {', '.join(sorted(intersection))}. The missing sample is: {missing}.")
                    
                    # 用户输入已有岩石类型的地质信息
                    ages = {}
                    lons = {}
                    lats = {}
                    for rock in sorted(intersection):
                        ages[rock] = st.number_input(f"Enter the age of {rock} (Ma)", min_value=0.0, value=0.0, format="%.2f", key=f"{rock}_age")
                        lons[rock] = st.number_input(f"Enter the longitude of {rock} (°)", format="%.2f", value=0.0, key=f"{rock}_lon")
                        lats[rock] = st.number_input(f"Enter the latitude of {rock} (°)", format="%.2f", value=0.0, key=f"{rock}_lat")

                    # 计算范围并检查条件
                    age_range = max(ages.values()) - min(ages.values())
                    lon_range = max(lons.values()) - min(lons.values())
                    lat_range = max(lats.values()) - min(lats.values())

                    # 检查年龄和坐标条件
                    if age_range <= 10 and lon_range <= 5 and lat_range <= 5:
                        st.success(f"According to the IBM regional rock sequence study, your study area may have an initial subduction event! However, one type of sample ({missing}) is missing. Please add it.")
                    else:
                        if age_range > 10:
                            st.warning("The age range of the samples varies greatly. Please consider the actual situation in your study area.")
                        if lon_range > 5 or lat_range > 5:
                            st.warning("The longitude and latitude ranges of the samples vary greatly. Please consider the actual situation in your study area.")

                # 情况 3：只检测到一种岩石类型
                elif len(intersection) == 1:
                    missing = list(target_classes - intersection)
                    st.info(f"Only one rock type was detected: {list(intersection)[0]}. Two samples are missing: {missing[0]} and {missing[1]}. Please add them.")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
