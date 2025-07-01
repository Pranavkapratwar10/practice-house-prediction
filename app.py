import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Title of app
st.title("🏡 California House Price Prediction")

# Load dataset
df = pd.read_csv(r"C:\downloads\kc_house_data.csv")

# Feature selection
x = df[['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'grade',
        'sqft_lot', 'view', 'yr_built', 'sqft_basement', 'lat', 'waterfront']]
y = df['price']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Train model
model = LinearRegression()
model.fit(X_train, Y_train)

# Streamlit Inputs
bedroom = st.slider("Select number of bedrooms", 0, 20, step=1)
bathroom = st.slider("Select number of bathrooms", 0, 10, step=1)
sqft_living = st.number_input("Enter sqft living area", min_value=100, max_value=10000, step=25)
floors = st.slider("Number of floors", 1, 5, step=1)
grade = st.slider("House grade", 1, 13, step=1)
sqft_lot = st.number_input("Enter sqft lot area", min_value=100, max_value=100000, step=50)
view = st.selectbox("Does it have a view?", [0, 1])
year_build = st.number_input("Year built", min_value=1900, max_value=2025, step=1)
sqft_basement = st.number_input("Enter basement sqft", min_value=0, max_value=5000, step=50)
lat = st.slider("Latitude", 47, 49)
waterfront = st.selectbox("Is it on waterfront?", [0, 1])

# Make prediction
if st.button("Predict House Price"):
    input_data = np.array([[bedroom, bathroom, sqft_living, floors, grade,
                            sqft_lot, int(view), year_build, sqft_basement, lat, int(waterfront)]])
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated House Price: ₹ {prediction:,.2f}")
