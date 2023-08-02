import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd
import xgboost

xgb_model = joblib.load('xgb_4f')

# Title
st.title("Used Car Price Prediction")

# Description
text = '''This app predicts the market price of a used car based on key attributes like mileage, age, make and model.
         Simply input the details of the car and hit 'Predict' to get an estimated price! 
         The prediction is made using a machine learning model trained on car sale price data.
         Read more about the project and model training in [this article](https://medium.com/@muhammet/building-a-dynamic-price-prediction-model-tackling-concept-drift-in-automobile-markets-a7b6c52112c2). 
         Try out the app below and see the model in action!
'''
st.markdown(text)

make_model = pd.read_csv('make_model.csv', index_col=0)

make_ = make_model.columns.tolist()

# User inputs
mileage = st.number_input("Mileage", min_value=0, max_value=100000, value=0)
year = st.number_input("Year", min_value=2000, max_value=2022, value=2000)
make = st.selectbox("Make", make_, placeholder="Toyota")
model = st.selectbox("Model", make_model[make].tolist())

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
    st.subheader(f'The predicted price is £{price[0]:.2f}')
    text = f'''
        The predicted price assumes the car is in average condition for {year} {make} {model}.
        Actual price may vary between £{price[0] - price[0]*0.1102:.2f} to £{price[0] + price[0]*0.1102:.2f} 
        depending on overall condition and any mechanical issues.
'''
    st.success(text)
