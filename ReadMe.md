# Used Car Price Prediction App
This is a simple web app built with Streamlit to predict used car prices based on key attributes like make, model, mileage, and year. It uses a pre-trained machine learning model to generate the price predictions.

## Demo
A live demo of the app is available here: 

## How it Works
The app takes in user inputs for mileage, year, make, and model. These are processed into a feature vector that is passed to a pre-trained regression model. The model then makes a single price prediction which is displayed to the user.

The model itself was trained on used car listing data scraped from eBay Kleinanzeigen. The training data includes details like make, model, year, mileage, and price for used car ads. This was used to fit a Random Forest Regressor model using Scikit-Learn.

The trained model is serialized with Pickle and loaded into the Streamlit app.

## Running the App Locally
To run the app locally, clone this repo and create a new conda environment:

git clone https://github.com/bayrakm/Car_price_prediction_app.git

conda create -n car-app-env python=3.10

Install dependencies:
requirements.txt

streamlit run app.py

The app will be available at http://localhost:8501. Adjust the code as needed to use your own model and data.

The app available online https://predictcarprice.streamlit.app/
