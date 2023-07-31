import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd

xgb_model = joblib.load('xgb_4f')

# Title
st.title("Used Car Price Prediction")

make_model = pd.read_csv('make_model.csv', index_col=0)

make_ = make_model.columns.tolist()

# User inputs
mileage = st.number_input("Mileage", min_value=0, max_value=1000000, value=10000)
year = st.number_input("Year", min_value=2000, max_value=2022, value=2000)
make = st.selectbox("Make", [''] + make_)
model = st.selectbox("Model", [''] + make_model[make].tolist())

# Create feature vector
features = [year, mileage]

sel_make = 'Make_' + make
sel_model = 'Model_1_' + model

make_features = xgb_model.feature_names_in_[2:6]
model_features = xgb_model.feature_names_in_.tolist()[6:]

for make_name in make_features:
    features.append(1 if sel_make == make_name else 0)
for model_name in model_features:
    features.append(1 if sel_model == model_name else 0)



features = np.array([features])

# Make prediction
if st.button("Predict"):
    price = xgb_model.predict(features)
    st.success(f"The predicted price is £{price[0]:.2f} with +/- £{price[0]*0.1102:.2f} price error range")
