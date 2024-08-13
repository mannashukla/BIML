# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston
import tensorflow as tf
from tensorflow import keras

# Load the Boston Housing dataset
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target

# View the first few rows
print(df.head())

# Exploratory Data Analysis (EDA)
# Summary statistics
print(df.describe())

# Correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Pair plot
sns.pairplot(df, x_vars=['RM', 'LSTAT', 'PTRATIO'], y_vars='PRICE', height=5, aspect=0.7)
plt.show()

# Data Preprocessing
# Check for missing values
print(df.isnull().sum())

# Scaling features
scaler = StandardScaler()
features = df.drop('PRICE', axis=1)
target = df['PRICE']
features_scaled = scaler.fit_transform(features)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Model Building with Scikit-learn (Random Forest)
from sklearn.ensemble import RandomForestRegressor

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluation
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest MSE: {mse_rf:.2f}")
print(f"Random Forest R2 Score: {r2_rf:.2f}")

# Model Building with TensorFlow
# Define the model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)

# Evaluate the model
loss, mse = model.evaluate(X_test, y_test)
print(f"TensorFlow Model MSE: {mse:.2f}")

# Predictions
y_pred_tf = model.predict(X_test).flatten()

# Evaluation
mse_tf = mean_squared_error(y_test, y_pred_tf)
r2_tf = r2_score(y_test, y_pred_tf)

print(f"TensorFlow Model MSE: {mse_tf:.2f}")
print(f"TensorFlow Model R2 Score: {r2_tf:.2f}")

# Compare both models
print(f"Random Forest R2 Score: {r2_rf:.2f}")
print(f"TensorFlow R2 Score: {r2_tf:.2f}")

# Visualizing Model Performance
# Plotting training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting predictions vs actual values
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_tf, alpha=0.6)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Predictions vs Actual Prices')
plt.show()

# Saving and Loading the Model
# Save the model
model.save('house_price_prediction_model.h5')

# Load the model
loaded_model = keras.models.load_model('house_price_prediction_model.h5')

# Verify the model is loaded correctly
loss, mse = loaded_model.evaluate(X_test, y_test)
print(f"Loaded Model MSE: {mse:.2f}")
