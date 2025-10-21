import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.mixture import GaussianMixture
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import io

st.set_page_config(page_title="Rock Type Prediction and Subduction Event Detection", layout="wide")
st.title("Rock Classification Prediction + Initial Subduction Event Identification")

# --------- 缓存训练数据和模型 ---------
@st.cache_resource
def load_data_and_models():
    train_file_path = "FAB-Boninite-HMA-IAT-CA.xlsx"
    train_data = pd.read_excel(train_file_path)

    X_train = train_data.drop(train_data.columns[0], axis=1)
    y_train = train_data.iloc[:, 0]

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # 训练集缺失值用均值填充，方便LDA等后续步骤
    X_train_filled = X_train.fillna(X_train.mean())

    # 定义模型
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

    hard_ensemble.fit(X_train_filled, y_train_encoded)
    soft_ensemble.fit(X_train_filled, y_train_encoded)

    return X_train, X_train_filled, y_train, y_train_encoded, label_encoder, hard_ensemble, soft_ensemble

X_train, X_train_filled, y_train, y_train_encoded, label_encoder, hard_ensemble, soft_ensemble = load_data_and_models()
st.success("✅ Training data and models loaded (cached).")

# ========== Upload Prediction File ==========
predict_file = st.file_uploader("Upload prediction data file (e.g., application.xlsx)", type=["xlsx"])
if predict_file:
    input_data = pd.read_excel(predict_file)
    st.success("✅ Prediction data loaded successfully")

    # Match column names
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

    st.subheader("Prediction Results")
    st.dataframe(input_data)

    # Download button
    output = io.BytesIO()
    input_data.to_excel(output, index=False, engine="openpyxl")
    output.seek(0)
    st.download_button(
        "Download Prediction Results",
        data=output,
        file_name="predicted_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # --------- SiO2-MgO 背景图绘制 ---------
    st.subheader("SiO2-MgO Background Plot of Prediction")
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
                ax.scatter(
                    sio2[class_indices], mgo[class_indices],
                    color=class_colors[class_name],
                    label=class_name,
                    alpha=0.7, s=150,
                    edgecolors='k',
                    zorder=1
                )

            ax.set_xlim(45, 70)
            ax.set_ylim(0, 25)
            ax.set_xlabel('SiO2')
            ax.set_ylabel('MgO')
            ax.set_title('SiO2-MgO Background Plot by Predicted Class')
            ax.legend()
            ax.grid(False)
            st.pyplot(fig)
        else:
            st.warning("❗ SiO2 or MgO column missing in input data, cannot plot.")
    except Exception as e:
        st.error(f"❌ Failed to generate SiO2-MgO plot: {e}")

    # --------- LDA降维与置信椭圆绘制 ---------
    st.subheader("LDA Projection with 95% Confidence Ellipses")

    from matplotlib.patches import Ellipse

    # 填补缺失值
    X_input_filled = X_input.fillna(X_train.mean())

    # 定义颜色映射
    category_colors = {
        'HMA': 'lightskyblue',
        'boninite': 'red',
        'CA': 'sandybrown',
        'IAT': 'green',
        'FAB': 'lightseagreen'
    }
    label_mapping = {label: name for label, name in enumerate(label_encoder.classes_)}

    lda = LDA(n_components=2)
    X_train_lda = lda.fit_transform(X_train_filled, y_train_encoded)
    X_input_lda = lda.transform(X_input_filled)

    fig1, ax1 = plt.subplots(figsize=(10, 8))

    # 训练数据散点，空心圆
    for class_idx in np.unique(y_train_encoded):
        class_name = label_mapping[class_idx]
        ax1.scatter(
            X_train_lda[y_train_encoded == class_idx, 0],
            X_train_lda[y_train_encoded == class_idx, 1],
            label=f"Train: {class_name}",
            alpha=0.3,
            color=category_colors.get(class_name, 'gray'),
            s=80,
            marker='o',
            facecolors='none',
            edgecolors=category_colors.get(class_name, 'gray')
        )

    # 用户数据散点，实心叉
    for class_name in np.unique(predicted_labels):
        idx = input_data["Predicted Class"] == class_name
        ax1.scatter(
            X_input_lda[idx, 0],
            X_input_lda[idx, 1],
            label=f"User: {class_name}",
            marker='X',
            s=200,
            edgecolor='k',
            color=category_colors.get(class_name, 'gray')
        )

    # 添加置信椭圆
    for class_idx in np.unique(y_train_encoded):
        class_data = X_train_lda[y_train_encoded == class_idx]
        gm = GaussianMixture(n_components=1).fit(class_data)
        mean = gm.means_[0]
        cov = gm.covariances_[0]
        v, w = np.linalg.eigh(cov)
        v = 2.4477 * np.sqrt(v)  # 95%置信区间
        angle = np.degrees(np.arctan2(w[0, 1], w[0, 0]))
        ell = Ellipse(
            xy=mean, width=v[0], height=v[1], angle=angle,
            edgecolor=category_colors.get(label_mapping[class_idx], 'gray'),
            facecolor='none', lw=2, alpha=0.7
        )
        ax1.add_patch(ell)

    ax1.set_xlabel("LDA Component 1")
    ax1.set_ylabel("LDA Component 2")
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-10, 10)
    ax1.legend(loc='best', fontsize=12)
    ax1.grid(False)
    ax1.set_title("LDA 2D Projection with 95% Confidence Ellipses")
    st.pyplot(fig1)

    # ========== Subduction Event Detection ==========
    st.subheader("Initial Subduction Event Detection")
    target_classes = {'FAB', 'boninite', 'HMA'}
    detected_classes = set(input_data['Predicted Class'].unique())
    intersection = detected_classes & target_classes

    if intersection:
        st.info(f"Detected key rock types: {', '.join(intersection)}")

        if intersection == target_classes:
            st.success("✅ All FAB, boninite, and HMA detected. Please input geological information:")

            with st.form("subduction_form"):
                ages, lons, lats = {}, {}, {}
                for rock in sorted(target_classes):
                    ages[rock] = st.number_input(f"{rock} Age (Ma)", step=0.1, key=f"{rock}_age")
                    lons[rock] = st.number_input(f"{rock} Longitude (°)", step=0.1, key=f"{rock}_lon")
                    lats[rock] = st.number_input(f"{rock} Latitude (°)", step=0.1, key=f"{rock}_lat")
                submitted = st.form_submit_button("Determine if initial subduction event occurred")

            if submitted:
                age_range = max(ages.values()) - min(ages.values())
                lon_range = max(lons.values()) - min(lons.values())
                lat_range = max(lats.values()) - min(lats.values())

                if age_range <= 10 and lon_range <= 5 and lat_range <= 5:
                    st.success("🎉 Possible initial subduction event detected! (Based on IBM rock sequence)")
                else:
                    if age_range > 10:
                        st.warning("⚠️ Age range is too wide. Check geological context.")
                    if lon_range > 5 or lat_range > 5:
                        st.warning("⚠️ Spatial range is too large. Check sample distribution.")
        else:
            missing = target_classes - intersection
            st.warning(f"⚠️ Missing key rock types: {', '.join(missing)}")
    else:
        st.error("❌ No FAB, boninite, or HMA detected.")
