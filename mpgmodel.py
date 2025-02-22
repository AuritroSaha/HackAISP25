import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load dataset (replace with actual file path)
df = pd.read_csv("dataGen.csv")

# Display the first few rows
print(df.head())

# Define the target variable (MPG)
target_column = "MPG"  # Change this if MPG has a different column name
X = df.drop(columns=[target_column])  # Features
y = df[target_column]  # Target

print(X)
print(y)