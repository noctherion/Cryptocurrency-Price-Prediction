import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Step 1: Data Collection (Assuming you have a CSV file with historical price and volume data)
df = pd.read_csv('cryptocurrency_data.csv')
print(df.head())

# Step 2: Data Preprocessing:
# Drop missing values and select relevant features
df = df.dropna()
X = df[['Volume', 'Open', 'High', 'Low']].values
y = df['Close'].values

# Step 3: Feature Engineering (None in this simplified example)

# Step 4: Model Building
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Model Training (Already done during model building)

# Step 6: Model Evaluation
# Evaluate the model's performance on the testing set
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

# Step 7: Make Predictions (Assuming you have new data for future prices)
future_data = np.array([[1000000, 7000, 8000, 6000]])  # Example new data
future_prediction = model.predict(future_data)

print("Predicted Future Price:", future_prediction[0])
