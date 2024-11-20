#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv('Bengaluru_House_Data.csv')


# In[3]:


data.head()


# In[4]:


data.shape


# In[6]:


data.info()


# In[10]:


for column in data.columns:
    print(data[column].value_counts())
    print('*'*20)


# In[12]:


data.isnull().sum()


# In[13]:


data.drop(columns = ['area_type', 'availability', 'society', 'balcony'], inplace= True)


# In[14]:


data.describe()


# In[15]:


data.info()


# In[17]:


data['location'].value_counts()


# In[18]:


data['location'] = data['location'].fillna('Sarjapur Road')


# In[19]:


data['size'].value_counts()


# In[20]:


data['size'] = data['size'].fillna('2BHK')


# In[21]:


data['bath'] = data['bath'].fillna(data['bath'].median())


# In[22]:


data.info()


# In[24]:


data['bhk'] = data['size'].str.extract('(\d+)').astype(int)


# In[25]:


data[data.bhk > 20]


# In[26]:


data['total_sqft'].unique()


# In[30]:


def convertRange(x):
    try:
        temp = str(x).split('_')  # Ensure x is treated as a string before splitting
        if len(temp) == 2:  # If it's a range with two values
            return (float(temp[0]) + float(temp[1])) / 2  # Calculate the average
        return float(x)  # Convert single numeric value to float
    except:
        return None  # Handle invalid cases by returning None


# In[31]:


data['total_sqft'] = data['total_sqft'].apply(convertRange)


# In[32]:


data.head()


# In[33]:


data['price_per_sqft'] = data['price'] * 100000/ data['total_sqft']


# In[34]:


data['price_per_sqft']


# In[35]:


data.describe()


# In[36]:


data['location'].value_counts()


# In[37]:


data['location'] = data['location'].apply(lambda x: x.strip())
location_count = data['location'].value_counts()


# In[38]:


location_count


# In[39]:


location_count_less_10 = location_count[location_count<=10]
location_count_less_10


# In[40]:


data['location'] = data['location'].apply(lambda x: 'other' if x in location_count_less_10 else x)


# In[41]:


data['location'].value_counts()


# # Outlier detection and removal

# In[42]:


data.describe()


# In[43]:


(data['total_sqft']/data['bhk']).describe


# In[45]:


data = data[((data['total_sqft']/data['bhk'])>= 300)]
data.describe()


# In[46]:


data.shape


# In[47]:


data.price_per_sqft.describe()


# In[49]:


def remove_outlier_sqft(df):
    df_output = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        
        gen_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_output = pd.concat([df_output, gen_df], ignore_index=True)
    return df_output
data = remove_outlier_sqft(data)
data.describe()


# In[50]:


def bhk_outlier_removar(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats = { 'mean' : np.mean(bhk_df.price_per_sqft), 'std': np.std(bhk_df.price_per_sqft), 'count': bhk_df.shape[0]
                                         }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')
                
        


# In[52]:


data = bhk_outlier_removar(data)
data


# In[53]:


data.shape


# In[54]:


data.drop(columns = ['size', 'price_per_sqft'], inplace = True)


# # Cleaned data

# In[55]:


data.head()


# In[56]:


data.to_csv('Cleaned_data.csv')


# In[57]:


x = data.drop(columns = ['price'])
y = data['price']


# In[67]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline


# In[60]:


X_train,X_test,y_train,Y_test = train_test_split(x,y, test_size=0.2, random_state=0)


# In[61]:


print(X_train.shape)
print(X_test.shape)


# # Applying Linear Regression

# In[63]:


column_trans = make_column_transformer((OneHotEncoder(sparse=False), ['location']), remainder= 'passthrough')


# In[64]:


scaler = StandardScaler()


# In[68]:


pipeline = Pipeline([
    ('preprocessor', column_trans),  # Apply column transformer
    ('scaler', StandardScaler()),  # Standardize features
    ('regressor', LinearRegression())  # Linear Regression model
])


# In[69]:


pipeline.fit(X_train, y_train)


# In[70]:


y_pred = pipeline.predict(X_test)


# In[72]:


r2_score(Y_test, y_pred)


# # Applying Lasso

# In[74]:


lasso = Lasso()


# In[75]:


pipeline = Pipeline([
    ('preprocessor', column_trans),  # Apply column transformer
    ('scaler', StandardScaler()),  # Standardize features
    ('regressor', Lasso(alpha=0.1))  # Lasso Regression with regularization parameter alpha
])


# In[76]:


pipeline.fit(X_train, y_train)


# In[78]:


y_pred_lasso = pipeline.predict(X_test)


# In[79]:


r2_score(Y_test, y_pred_lasso)


# In[82]:


import pickle


# In[84]:


pickle.dump(pipeline, open('LassoModel.pkl', 'wb'))


# In[86]:


from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('LassoModel.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Lasso Model is live!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Assuming input is a list of feature values
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)


# In[87]:


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




