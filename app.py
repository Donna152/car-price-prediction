import streamlit as st
import pandas as pd
import pickle

# 1. Load the pipeline and the dataframe from .pkl files
try:
    pipe = pickle.load(open('LinearRegression.pkl', 'rb'))
    df = pickle.load(open('car.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# 2. Page Title
st.title("Welcome to Car Price Predictor")

# 3. Dynamic Inputs
# Pulling unique values directly from the unpickled dataframe
company = st.selectbox("Select Company:", sorted(df['company'].unique()))

# Filter models based on the selected company
models_in_company = df[df['company'] == company]['name'].unique()
model = st.selectbox("Select Model:", sorted(models_in_company))

year = st.selectbox("Select Year of Purchase:", sorted(df['year'].unique(), reverse=True))
fuel_type = st.selectbox("Select Fuel Type:", sorted(df['fuel_type'].unique()))
kms_driven = st.number_input("Enter Number of Kilometers travelled:", min_value=0, step=100)

# 4. Prediction Logic
if st.button("Predict Price"):
    try:
        # Create input DataFrame
        input_data = pd.DataFrame(
            [[company, model, year, kms_driven, fuel_type]],
            columns=['company', 'name', 'year', 'kms_driven', 'fuel_type']
        )

        # Predict
        prediction = pipe.predict(input_data)

        # Display result
        st.success(f"The predicted price is: ₹{prediction[0]:,.2f}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")