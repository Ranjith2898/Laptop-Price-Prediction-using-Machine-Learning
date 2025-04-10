# Laptop-Price-Prediction-using-Machine-Learning
Predict laptop prices using machine learning based on specs like RAM, SSD, CPU, GPU, brand, and more. Achieved 88.6% accuracy using Gradient Boosting. Includes EDA, feature engineering, model tuning, and a Streamlit web app for real-time price predictions. 
## Overview

This project aims to predict the prices of laptops using machine learning models based on various hardware specifications and brand features. It was developed as part of OdinSchool's Data Science capstone project.

## Dataset

The dataset contains details of 1300+ laptops with attributes like:
- Company
- Type
- Screen Size & Resolution
- CPU & GPU
- RAM & Storage
- Weight
- Operating System
- Price (target)

## Key Features

- Data preprocessing and feature engineering (e.g., PPI, storage split, CPU/GPU brands)
- Exploratory Data Analysis (EDA)
- Multiple ML models: Linear Regression, Decision Tree, Random Forest, Gradient Boosting
- Best Model: GradientBoostingRegressor with 88.6% accuracy
- Model serialization with `pickle`
- Streamlit web app for user-friendly predictions

## How to Run

1. Clone the repo
2. Install requirements: `pip install -r requirements.txt`
3. Run the Streamlit app:  
   ```bash
   streamlit run app.py
   ```

## Files

- `notebook.ipynb`: Full analysis and modeling
- `df.pkl`, `pipe.pkl`: Serialized model and data
- `app.py`: Streamlit app code
- `requirements.txt`: Dependencies
- `README.md`: Project documentation

## Future Enhancements

- Integrate real-time product listings
- Improve model with newer data
- Enhance the UI for user interaction

---

Created with passion for Data Science.


# app.py

import streamlit as st
import pickle
import numpy as np

# Load model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Price Predictor")

# Input fields
company = st.selectbox('Brand', df['Company'].unique())
typename = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM (in GB)', sorted(df['Ram'].unique()))
weight = st.number_input('Weight of Laptop (kg)')
touchscreen = st.selectbox('Touchscreen', ['Yes', 'No'])
ips = st.selectbox('IPS Display', ['Yes', 'No'])
ppi = st.number_input('PPI (Pixels Per Inch)')
cpu = st.selectbox('CPU Brand', df['Cpu brand'].unique())
hdd = st.number_input('HDD (in GB)')
ssd = st.number_input('SSD (in GB)')
gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique())
os = st.selectbox('Operating System', df['os'].unique())

# Predict button
if st.button('Predict Price'):
    query = np.array([[company, typename, ram, weight,
                       1 if touchscreen == 'Yes' else 0,
                       1 if ips == 'Yes' else 0,
                       ppi, cpu, hdd, ssd, gpu, os]])
    prediction = pipe.predict(query)[0]
    st.title(f"Predicted Price: ₹{np.exp(prediction):.2f}")

