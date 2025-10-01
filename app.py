import streamlit as st
import pickle
import pandas as pd
import numpy as np

# =======================
# Load trained artifacts
# =======================
with open(r"C:/Users/FlavyaLekkala/laptop_project/laptop_price_model.pkl", "rb") as f:
    artifacts = pickle.load(f)

model = artifacts["model"]
scaler = artifacts["scaler"]
label_encoders = artifacts["label_encoders"]
features = artifacts["features"]
categorical_cols = ['Brand', 'Processor', 'OS', 'Storage_Type']

numeric_cols = ["RAM", "Storage_Size", "Generation"]



# =======================
# Streamlit App
# =======================
st.title("💻 Laptop Price Prediction")
st.write("Fill the details below and get the predicted price of a laptop.")



# =======================
# User inputs with placeholders
# =======================
brand_list = ['Lenovo','ASUS','HP','DELL','Acer','MSI','Infinix','Apple','Samsung','MOTOROLA','GIGABYTE','Colorful']
brand = st.selectbox("Brand", ["Select Brand"] + brand_list)
brand = None if brand == "Select Brand" else brand

processor_list = ['Intel Core i5','Intel Core i3','Intel','Intel Core i7','AMD Ryzen 5','AMD Ryzen 7','AMD Ryzen 3',
                  'MediaTek','Intel Core i9','Apple M3','AMD Ryzen 9','Apple M2','Apple M4']
processor = st.selectbox("Processor", ["Select Processor"] + processor_list)
processor = None if processor == "Select Processor" else processor

os_list = ['Windows 11','Windows 10','Chrome OS','macOS','Mac OS','Windows 1']
os = st.selectbox("Operating System", ["Select OS"] + os_list)
os = None if os == "Select OS" else os

storage_type_list = ["SSD","EMMC"]
storage_type = st.selectbox("Storage Type", ["Select Storage Type"] + storage_type_list)
storage_type = None if storage_type == "Select Storage Type" else storage_type

# Dynamic Generation based on processor
valid_generations = None
if processor:
    if "Intel" in processor:
        valid_generations = [1, 10, 11, 12, 13, 14]
    elif "Ryzen" in processor:
        valid_generations = [3, 5, 7, 9]
    elif "Apple" in processor:
        valid_generations = [2, 3, 4]
    elif "MediaTek" in processor:
        valid_generations = [520, 528, 1200]

generation = None
if valid_generations:
    generation = st.selectbox("Processor Generation", ["Select Generation"] + valid_generations)
    generation = None if generation == "Select Generation" else generation

# Hardcoded RAM values
ram_values = [8, 16, 32, 64, 128]
ram = st.selectbox("RAM (GB)", ["Select RAM"] + ram_values)
ram = None if ram == "Select RAM" else ram

# Hardcoded storage sizes
storage_values = [32, 64, 128, 256, 512, 1024]
storage_size = st.selectbox("Storage Size (GB)", ["Select Storage"] + storage_values)
storage_size = None if storage_size == "Select Storage" else storage_size



# =======================
# Predict Button
# =======================
if st.button("Predict Price"):
    # Check if any value is missing
    if None in [brand, processor, os, storage_type, generation, ram, storage_size]:
        st.warning("❌ Please fill all the fields before predicting.")
    else:
        # Collect user input into DataFrame
        new_laptop = {
            'Brand': brand,
            'Processor': processor,
            'OS': os,
            'Storage_Type': storage_type,
            'RAM': ram,
            'Storage_Size': storage_size,
            'Generation': generation
        }
        new_df = pd.DataFrame([new_laptop])

        # =======================
        # Encode categorical columns
        # =======================
        for col in categorical_cols:
            le = label_encoders[col]
            new_df[col] = le.transform(new_df[col])

        # =======================
        # Scale Numeric Columns
        # =======================

        scaled_numeric = scaler.transform(new_df[numeric_cols].to_numpy())
        new_df[numeric_cols] = scaled_numeric


        # Ensure correct feature order
        new_df = new_df[features]

        # =======================
        # Prediction
        # =======================
        predicted_price = model.predict(new_df)[0]
        st.success(f"💰 Predicted Laptop Price: ₹ {predicted_price:,.0f}")

        


