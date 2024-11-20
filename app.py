#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('LassoModel.pkl', 'rb'))

# App title
st.title("Lasso Regression Model")

# Input features
location = st.text_input("Enter location:")
sqft = st.number_input("Total square feet:")
bhk = st.slider("Number of BHK", 1, 10)
bath = st.slider("Number of bathrooms", 1, 10)

# Make predictions
if st.button("Predict"):
    # Replace this with proper feature preparation
    features = np.array([location, sqft, bhk, bath]).reshape(1, -1)
    prediction = model.predict(features)
    st.success(f"Predicted Price: {prediction[0]}")


# In[ ]:




