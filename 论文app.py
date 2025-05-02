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

# 构建集成模型（XGB + ET + LGBM）
xgb = XGBClassifier(n_estimators=100, eval_metric='mlogloss', random_state=42)  # 移除 use_label_encoder 参数
et = ExtraTreesClassifier(n_estimators=100, random_state=42)
lgbm = LGBMClassifier(n_estimators=100, random_state=42)  # 移除 use_label_encoder 参数

model = VotingClassifier(estimators=[
    ('xgb', xgb),
    ('et', et),
    ('lgbm', lgbm)
], voting='hard')

model.fit(X_train, y_train_encoded)

# 4. 文件上传
uploaded_file = st.file_uploader("Upload a new Excel file for prediction", type=["xlsx"])
if uploaded_file is not None:
    try:
        input_data = pd.read_excel(uploaded_file)
        st.write("File uploaded successfully!")
        st.write(input_data.head())  # Print the first few rows of the uploaded data for inspection

        # 数据预处理
        X_input = input_data.drop(input_data.columns[0], axis=1)  # Assuming the first column is the target column
        X_input = X_input.fillna(0)  # Fill missing values with 0
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

            unique_classes = np.unique(predicted_classes)
            cmap = plt.get_cmap('tab10')
            class_colors = {class_name: cmap(i) for i, class_name in enumerate(unique_classes)}

            # Plot each class's points within the defined coordinate range
            for class_name in unique_classes:
                class_indices = predicted_classes == class_name
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

    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
else:
    st.info("Please upload a valid Excel file to start predictions.")

# 模型加载完成信息
st.success("The ensemble model (XGB + ET + LGBM) has been loaded and trained successfully.")
