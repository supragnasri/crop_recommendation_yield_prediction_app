import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Load datasets
# -----------------------------
crop_data = pd.read_csv("Crop_recommendation.csv")
yield_data = pd.read_csv(r"C:\Users\SUPRAGNA SRI\ML_Project_1\crop_yield.csv")

# -----------------------------
# CROP RECOMMENDATION MODEL
# -----------------------------

X_crop = crop_data.drop("label", axis=1)
y_crop = crop_data["label"]

X_train_crop, X_test_crop, y_train_crop, y_test_crop = train_test_split(
    X_crop, y_crop, test_size=0.2, random_state=42
)

rf_crop = RandomForestClassifier()
dt_crop = DecisionTreeClassifier()
knn_crop = KNeighborsClassifier()

rf_crop.fit(X_train_crop, y_train_crop)
dt_crop.fit(X_train_crop, y_train_crop)
knn_crop.fit(X_train_crop, y_train_crop)

# Predictions for accuracy comparison
y_pred_rf_crop = rf_crop.predict(X_test_crop)
y_pred_dt_crop = dt_crop.predict(X_test_crop)
y_pred_knn_crop = knn_crop.predict(X_test_crop)

# Best crop model
crop_model = rf_crop

# -----------------------------
# YIELD PREDICTION MODEL
# -----------------------------

if "Unnamed: 0" in yield_data.columns:
    yield_data = yield_data.drop(columns=["Unnamed: 0"])

item_encoder = LabelEncoder()
area_encoder = LabelEncoder()

yield_data["Item"] = item_encoder.fit_transform(yield_data["Item"])
yield_data["Area"] = area_encoder.fit_transform(yield_data["Area"])

X_yield = yield_data.drop("hg/ha_yield", axis=1)
y_yield = yield_data["hg/ha_yield"]

X_train_y, X_test_y, y_train_y, y_test_y = train_test_split(
    X_yield, y_yield, test_size=0.2, random_state=42
)

rf_yield = RandomForestRegressor()
dt_yield = DecisionTreeRegressor()
lr_yield = LinearRegression()

rf_yield.fit(X_train_y, y_train_y)
dt_yield.fit(X_train_y, y_train_y)
lr_yield.fit(X_train_y, y_train_y)

rf_pred_y = rf_yield.predict(X_test_y)
dt_pred_y = dt_yield.predict(X_test_y)
lr_pred_y = lr_yield.predict(X_test_y)

yield_model = rf_yield

# -----------------------------
# STREAMLIT UI
# -----------------------------

st.title("🌾 Crop Recommendation & Yield Prediction System")

st.header("Enter Soil and Weather Conditions")

N = st.number_input("Nitrogen")
P = st.number_input("Phosphorus")
K = st.number_input("Potassium")

temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")
ph = st.number_input("Soil pH")
rainfall = st.number_input("Rainfall")

st.header("Yield Prediction Inputs")

area = st.number_input("Area Code (numeric)")
year = st.number_input("Year", value=2013)
pesticides = st.number_input("Pesticides Used")
avg_temp = st.number_input("Average Temperature")

# -----------------------------
# Prediction Function
# -----------------------------

def predict_crop_and_yield(N, P, K, temperature, humidity, ph, rainfall,
                           area, year, pesticides, avg_temp):

    crop_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    crop_prediction = crop_model.predict(crop_input)[0]

    st.success(f"Recommended Crop: {crop_prediction}")

    yield_crops = [c.lower() for c in item_encoder.classes_]

    if crop_prediction.lower() not in yield_crops:
        st.warning("Crop not present in yield dataset. Yield prediction unavailable.")
        return

    crop_encoded = item_encoder.transform(
        [item_encoder.classes_[yield_crops.index(crop_prediction.lower())]]
    )[0]

    yield_input = np.array([[area, crop_encoded, year, rainfall, pesticides, avg_temp]])

    predicted_yield = yield_model.predict(yield_input)[0]

    st.success(f"Predicted Yield: {predicted_yield}")

# -----------------------------
# Button
# -----------------------------

if st.button("Predict Crop and Yield"):

    predict_crop_and_yield(
        N, P, K, temperature, humidity, ph, rainfall,
        area, year, pesticides, avg_temp
    )

# -----------------------------
# Model Performance Display
# -----------------------------

st.sidebar.title("Model Performance")

st.sidebar.write("Crop Model Accuracy")

st.sidebar.write(
    "Random Forest:",
    accuracy_score(y_test_crop, y_pred_rf_crop)
)

st.sidebar.write(
    "Decision Tree:",
    accuracy_score(y_test_crop, y_pred_dt_crop)
)

st.sidebar.write(
    "KNN:",
    accuracy_score(y_test_crop, y_pred_knn_crop)
)

st.sidebar.write("")

st.sidebar.write("Yield Model Error (MSE)")

st.sidebar.write(
    "Random Forest:",
    mean_squared_error(y_test_y, rf_pred_y)
)

st.sidebar.write(
    "Decision Tree:",
    mean_squared_error(y_test_y, dt_pred_y)
)

st.sidebar.write(
    "Linear Regression:",
    mean_squared_error(y_test_y, lr_pred_y)
)