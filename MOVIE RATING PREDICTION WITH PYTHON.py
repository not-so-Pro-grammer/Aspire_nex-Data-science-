import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'Aspire Nex DS\MOVIE RATING PREDICTION WITH PYTHON\IMDb Movies India.csv'
try:
    movies_df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    movies_df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Data Preprocessing
movies_df = movies_df.dropna()
label_encoders = {}
categorical_columns = ['Genre', 'Director', 'Actors']

for column in categorical_columns:
    le = LabelEncoder()
    movies_df[column] = le.fit_transform(movies_df[column])
    label_encoders[column] = le

# Feature Selection
features = ['Genre', 'Director', 'Actors']
target = 'Rating'

X = movies_df[features]
y = movies_df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
