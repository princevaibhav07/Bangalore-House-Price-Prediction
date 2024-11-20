# 🏠 Bangalore House Price Prediction Model  

This project aims to predict house prices in Bangalore based on features such as location, square footage, number of bedrooms (BHK), and bathrooms. The solution includes robust data cleaning, feature engineering, machine learning model development, and a deployment-ready application.

---

## 🔍 Project Overview  
- **Goal**: Predict house prices using machine learning models.  
- **Features Used**: Location, total square feet, number of bedrooms (BHK), and bathrooms.  
- **Tech Stack**: Python, Pandas, NumPy, Scikit-learn, Flask, Streamlit.  

---

## 📁 Dataset  
- **Source**: `Bengaluru_House_Data.csv`  
- **Key Columns**:  
  - `location`: Property location.  
  - `size`: Number of bedrooms and bathrooms.  
  - `total_sqft`: Total area in square feet.  
  - `price`: Price of the property in lakhs.  

---

## 📊 Workflow  

1. **Data Cleaning**:  
   - Dropped unnecessary columns like `area_type`, `availability`, and `society`.  
   - Handled missing values in columns like `location`, `size`, and `bath`.  
   - Converted `size` into numerical `BHK` and `total_sqft` into a usable numerical format.  

2. **Feature Engineering**:  
   - Created new features like `price_per_sqft`.  
   - Consolidated locations with fewer data points into an "other" category.  

3. **Outlier Detection & Removal**:  
   - Used statistical methods to detect and remove outliers in `price_per_sqft`.  
   - Removed unrealistic data points based on property size and price.  

4. **Modeling**:  
   - Explored **Linear Regression** and **Lasso Regression** models.  
   - Achieved optimal performance using **Lasso Regression** with regularization.  


