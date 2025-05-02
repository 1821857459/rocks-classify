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

# Page Configuration
st.set_page_config(page_title="Rock Classification Prediction", layout="wide")

# 1. Load Training Data
train_file_path = r"FAB-Boninite-HMA-IAT-CA.xlsx"
train_data = pd.read_excel(train_file_path)

# 2. Data Preprocessing
X_train = train_data.drop(train_data.columns[0], axis=1)
y_train = train_data.iloc[:, 0]

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# 3. Build Ensemble Model (XGB + ET + LGBM)
xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
et = ExtraTreesClassifier(n_estimators=100, random_state=42)
lgbm = LGBMClassifier(n_estimators=100, random_state=42)

model = VotingClassifier(estimators=[
    ('xgb', xgb),
    ('et', et),
    ('lgbm', lgbm)
], voting='hard')

model.fit(X_train, y_train_encoded)

# 4. File Upload
uploaded_file = st.file_uploader("Upload a new Excel file for prediction", type=["xlsx"])
if uploaded_file is not None:
    input_data = pd.read_excel(uploaded_file)

    # Match Column Names
    matching_columns = {}
    for col_train in X_train.columns:
        for col_input in input_data.columns:
            if col_input.lower().startswith(col_train.lower()):
                matching_columns[col_train] = col_input
                break
        if col_train not in matching_columns:
            matching_columns[col_train] = None

    # Construct Input Features
    X_input = pd.DataFrame()
    for col_train, col_input in matching_columns.items():
        if col_input is not None:
            X_input[col_train] = input_data[col_input]
        else:
            X_input[col_train] = 0

    # Handle Missing Values
    if X_input.isnull().values.any():
        st.warning("Some values are missing in the uploaded data. Missing values have been filled with 0.")
        X_input = X_input.fillna(0)

    # 5. Prediction
    predicted_classes = model.predict(X_input)
    predicted_classes = label_encoder.inverse_transform(predicted_classes)

    # Use one of the models (e.g. LGBM) to calculate confidence
    confidence_model = lgbm
    confidence_model.fit(X_train, y_train_encoded)
    predicted_probabilities = confidence_model.predict_proba(X_input)
    confidence_scores = np.max(predicted_probabilities, axis=1)

    # Add Prediction and Confidence to Data
    input_data['Predicted Class'] = predicted_classes
    input_data['Confidence'] = confidence_scores

    # Display Prediction Results
    st.write(input_data)

    # Download Button for Results
    to_write = io.BytesIO()
    input_data.to_excel(to_write, index=False, engine='xlsxwriter')
    st.download_button("Download Prediction Results", data=to_write.getvalue(), file_name="predicted_results.xlsx")

    # 6. SiO2-MgO Scatter Plot with Background Image

    img_path = r"MgO-SiO2.jpg"

    try:
        img = Image.open(img_path)

        if 'SiO2' not in input_data.columns or 'MgO' not in input_data.columns:
            st.error("The uploaded Excel file is missing 'SiO2' or 'MgO' columns.")
        else:
            sio2 = input_data['SiO2']
            mgo = input_data['MgO']

            plt.figure(figsize=(10, 10))
            plt.imshow(img, extent=[45, 70, 0, 25], aspect='auto', zorder=0)

            # Use label_encoder's class order to ensure consistent colors
            unique_classes = label_encoder.classes_
            cmap = plt.get_cmap('tab10')
            class_colors = {class_name: cmap(i) for i, class_name in enumerate(unique_classes)}

            for class_name in unique_classes:
                class_indices = input_data['Predicted Class'] == class_name
                plt.scatter(sio2[class_indices], mgo[class_indices],
                            color=class_colors[class_name], label=class_name,
                            alpha=0.8, s=150, edgecolor='black', linewidth=0.5, zorder=1)

            plt.xlim(45, 70)
            plt.ylim(0, 25)
            plt.xlabel('SiO₂', fontsize=14)
            plt.ylabel('MgO', fontsize=14)
            plt.title('SiO₂-MgO Classification Plot (with Background Image)', fontsize=16)
            plt.legend(title="Predicted Class", loc='upper right', fontsize=12)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            st.pyplot(plt)

    except FileNotFoundError:
        st.error("Background image 'MgO-SiO2.jpg' not found. Please check the file path.")
    except Exception as e:
        st.error(f"Failed to generate the plot: {e}")

# Model Loading Complete Message
st.success("The ensemble model (XGB + ET + LGBM) has been successfully loaded and trained.")

# Check if FAB, boninite, HMA rock types are in the predicted results
detected_classes = set(input_data['Predicted Class'].unique())
target_classes = {'FAB', 'boninite', 'HMA'}
intersection = detected_classes & target_classes

# If any of the target rock types are detected
if intersection:
    # Ask user if they want to perform follow-up subduction event judgment
    if st.radio("Samples consistent with the initial subduction rock sequence were detected. Do you want to perform a follow-up judgment?", ["YES", "NO"]) == "YES":

        # Case 1: All three rock types appear
        if intersection == target_classes:
            st.success("Three types of rocks (FAB, boninite, and HMA) were detected. Please enter their geological information.")

            ages = {}
            lons = {}
            lats = {}

            # User inputs age, longitude, and latitude for each rock type
            for rock in sorted(target_classes):
                ages[rock] = st.number_input(f"Enter the age of {rock} (Ma)", min_value=0.0, value=0.0, format="%.2f", key=f"{rock}_age")
                lons[rock] = st.number_input(f"Enter the longitude of {rock} (°)", format="%.2f", value=0.0, key=f"{rock}_lon")
                lats[rock] = st.number_input(f"Enter the latitude of {rock} (°)", format="%.2f", value=0.0, key=f"{rock}_lat")

            # Calculate the range of ages and coordinates
            age_range = max(ages.values()) - min(ages.values())
            lon_range = max(lons.values()) - min(lons.values())
            lat_range = max(lats.values()) - min(lats.values())

            # Check if the time and space constraints are met
            if age_range <= 10 and lon_range <= 5 and lat_range <= 5:
                st.success("According to the IBM regional rock sequence study, your study area may have an initial subduction event!")
            else:
                if age_range > 10:
                    st.warning("The age range of the samples is too large. Please consider the actual situation in your study area.")
                if lon_range > 5 or lat_range > 5:
                    st.warning("The longitude and latitude ranges of the samples vary greatly. Please consider the actual situation in your study area.")

        # Case 2: Only two rock types appear
        elif len(intersection) == 2:
            missing = list(target_classes - intersection)[0]  # Identify the missing rock type
            st.info(f"Two rock types were detected: {', '.join(sorted(intersection))}. The missing sample is: {missing}.")

            ages = {}
            lons = {}
            lats = {}

            # User inputs geological information for the two detected rock types
            for rock in sorted(intersection):
                ages[rock] = st.number_input(f"Enter the age of {rock} (Ma)", min_value=0.0, value=0.0, format="%.2f", key=f"{rock}_age")
                lons[rock] = st.number_input(f"Enter the longitude of {rock} (°)", format="%.2f", value=0.0, key=f"{rock}_lon")
                lats[rock] = st.number_input(f"Enter the latitude of {rock} (°)", format="%.2f", value=0.0, key=f"{rock}_lat")

            age_range = max(ages.values()) - min(ages.values())
            lon_range = max(lons.values()) - min(lons.values())
            lat_range = max(lats.values()) - min(lats.values())

            # Check age and coordinate conditions
            if age_range <= 10 and lon_range <= 5 and lat_range <= 5:
                st.success(f"According to the IBM regional rock sequence study, your study area may have an initial subduction event! However, one type of sample ({missing}) is missing. Please add it.")
            else:
                if age_range > 10:
                    st.warning("The age range of the samples varies greatly. Please consider the actual situation in your study area.")
                if lon_range > 5 or lat_range > 5:
                    st.warning("The longitude and latitude ranges of the samples vary greatly. Please consider the actual situation in your study area.")

        # Case 3: Only one rock type appears
        elif len(intersection) == 1:
            missing = list(target_classes - intersection)
            st.info(f"Only one rock type was detected: {list(intersection)[0]}. Two samples are missing: {missing[0]} and {missing[1]}. Please add them.")
