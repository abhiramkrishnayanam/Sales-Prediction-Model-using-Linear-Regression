#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib



# In[2]:


# Load dataset
df = pd.read_csv(r"D:\important\datascience\BigMart Sales Data.csv")




# In[3]:


# Data exploration
print(df.head())  # Show first few rows


# In[4]:


print(df.describe())  # Summary statistics


# In[5]:


df['Item_Outlet_Sales'].hist(bins=70)  # Sales distribution


# In[6]:


print(df.dtypes)  # Check data types


# In[7]:


# Handling missing values
df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)


# In[8]:


df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0], inplace=True)


# In[9]:


# Feature selection
X = df[['Item_Weight', 'Item_Visibility', 'Item_MRP',
        'Outlet_Establishment_Year', 'Outlet_Size',
        'Outlet_Location_Type', 'Outlet_Type']]
y = df['Item_Outlet_Sales']


# In[10]:


# One-hot encoding of categorical variables
X = pd.get_dummies(X, drop_first=True)



# In[11]:


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# In[12]:


# Check for missing values
print(X_train.isnull().sum())  # Check for missing values in the features
print(y_train.isnull().sum())  # Check for missing values in the target variable



# In[16]:


# Train the model
model.fit(X_train, y_train)



# In[18]:


# Create a Linear Regression model
model = LinearRegression()




# In[19]:


# Train the model
model.fit(X_train, y_train)


# In[20]:


# Make predictions
y_pred = model.predict(X_test)


# In[22]:


# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)


# In[23]:


# Print the metrics
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R²): {r2}')


# In[24]:


# Plotting actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)  # Perfect prediction line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()


# In[25]:


# Get feature names and coefficients
feature_names = X.columns
coefficients = model.coef_


# In[26]:


# Create a DataFrame for visualization
importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
importance = importance.sort_values(by='Coefficient', ascending=False)


# In[28]:


# Print feature importance
print(importance)



# In[50]:


y = df['Item_Outlet_Sales']


# In[29]:


# Save the model to a file
joblib.dump(model, 'linear_regression_model.pkl')


# In[70]:





# In[53]:





# In[54]:





# In[ ]:





# In[56]:





# In[57]:





# In[58]:





# In[59]:





# In[60]:





# In[61]:





# In[62]:





# In[ ]:





# In[ ]:





# In[65]:





# In[ ]:





# In[67]:





# In[68]:





# In[ ]:





# In[71]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[77]:





# In[ ]:





# In[31]:





# In[ ]:





# In[ ]:




