import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

st.set_page_config(page_title="岩石分类预测与俯冲事件识别", layout="wide")
st.title("🌋 岩石分类预测 + 初始俯冲事件判定")

# ========== 加载训练数据 ==========
train_file_path = "FAB-Boninite-HMA-IAT-CA.xlsx"
train_data = pd.read_excel(train_file_path)
X_train = train_data.drop(train_data.columns[0], axis=1)
y_train = train_data.iloc[:, 0]

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

model_et = ExtraTreesClassifier(n_estimators=100, random_state=42)
model_xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
model_lgbm = LGBMClassifier(random_state=42)

hard_ensemble = VotingClassifier(
    estimators=[('et', model_et), ('xgb', model_xgb), ('lgbm', model_lgbm)],
    voting='hard'
)
soft_ensemble = VotingClassifier(
    estimators=[('et', model_et), ('xgb', model_xgb), ('lgbm', model_lgbm)],
    voting='soft'
)

hard_ensemble.fit(X_train, y_train_encoded)
soft_ensemble.fit(X_train, y_train_encoded)
st.success("✅ 模型加载并训练完成（使用本地训练集）")

# ========== 上传预测文件 ==========
predict_file = st.file_uploader("📂 上传预测数据文件（例如：应用.xlsx）", type=["xlsx"])
if predict_file:
    input_data = pd.read_excel(predict_file)
    st.success("✅ 预测数据读取成功")

    # 匹配列名
    matching_columns = {}
    processed_train_columns = [col.lower().strip() for col in X_train.columns]
    processed_input_columns = [col.lower().strip() for col in input_data.columns]

    for col_train, processed_col_train in zip(X_train.columns, processed_train_columns):
        for col_input, processed_col_input in zip(input_data.columns, processed_input_columns):
            if processed_col_input.startswith(processed_col_train):
                matching_columns[col_train] = col_input
                break
        else:
            matching_columns[col_train] = None

    X_input = pd.DataFrame()
    for col_train, col_input in matching_columns.items():
        X_input[col_train] = input_data[col_input] if col_input else 0

    predicted_classes = hard_ensemble.predict(X_input)
    probs = soft_ensemble.predict_proba(X_input)
    confidences = np.max(probs, axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_classes)

    input_data["Predicted Class"] = predicted_labels
    input_data["Confidence"] = confidences

    st.subheader("📊 预测结果")
    st.dataframe(input_data)

    st.download_button("📥 下载预测结果", data=input_data.to_excel(index=False), file_name="predicted_results.xlsx")

    # ========== 背景图散点图 ==========
    st.subheader("🧪 SiO2-MgO 背景图预测分布")
    try:
        img = Image.open("MgO-SiO2.jpg")
        if 'SiO2' in input_data.columns and 'MgO' in input_data.columns:
            sio2 = input_data['SiO2']
            mgo = input_data['MgO']

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(img, extent=[45, 70, 0, 25], aspect='auto', zorder=0)

            unique_classes = np.unique(predicted_labels)
            cmap = plt.get_cmap('tab10')
            class_colors = {class_name: cmap(i) for i, class_name in enumerate(unique_classes)}

            for class_name in unique_classes:
                class_indices = predicted_labels == class_name
                ax.scatter(sio2[class_indices], mgo[class_indices],
                           color=class_colors[class_name], label=class_name,
                           alpha=0.7, s=150, edgecolor='k', zorder=1)

            ax.set_xlim(45, 70)
            ax.set_ylim(0, 25)
            ax.set_xlabel('SiO2')
            ax.set_ylabel('MgO')
            ax.set_title('SiO2-MgO Background Plot by Predicted Class')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.warning("❗ 输入数据缺少 SiO2 或 MgO 列，无法绘图")
    except Exception as e:
        st.error(f"❌ 图像绘制失败：{e}")

    # ========== 初始俯冲事件判定 ==========
    st.subheader("🧭 初始俯冲事件判定")
    target_classes = {'FAB', 'boninite', 'HMA'}
    detected_classes = set(input_data['Predicted Class'].unique())
    intersection = detected_classes & target_classes

    if intersection:
        st.info(f"🔍 检测到关键岩石类型：{', '.join(intersection)}")

        if intersection == target_classes:
            st.success("✅ 检测到 FAB、boninite 和 HMA，请输入地质信息：")
            ages, lons, lats = {}, {}, {}

            for rock in sorted(target_classes):
                ages[rock] = st.number_input(f"{rock} 年龄 (Ma)", step=0.1)
                lons[rock] = st.number_input(f"{rock} 经度 (°)", step=0.1)
                lats[rock] = st.number_input(f"{rock} 纬度 (°)", step=0.1)

            if st.button("🚀 判定区域是否存在初始俯冲事件"):
                age_range = max(ages.values()) - min(ages.values())
                lon_range = max(lons.values()) - min(lons.values())
                lat_range = max(lats.values()) - min(lats.values())

                if age_range <= 10 and lon_range <= 5 and lat_range <= 5:
                    st.success("🎉 区域可能存在初始俯冲事件！（基于 IBM 岩石序列）")
                else:
                    if age_range > 10:
                        st.warning("⚠️ 年龄跨度较大，请检查地质背景")
                    if lon_range > 5 or lat_range > 5:
                        st.warning("⚠️ 经纬度跨度较大，请检查样品分布")
        else:
            missing = target_classes - intersection
            st.warning(f"⚠️ 缺失关键类型：{', '.join(missing)}")
    else:
        st.error("❌ 未检测到 FAB、boninite 或 HMA")
