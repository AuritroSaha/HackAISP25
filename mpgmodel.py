import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load dataset (replace with actual file path)
df = pd.read_csv("Dataset/dataGen.csv")

# Display the first few rows
print(df.head())

# Define the target variable (MPG)
target_column = "MPG"  # Change this if MPG has a different column name
X = df.drop(columns=[target_column])  # Features
y = df[target_column]  # Target

print(X)
print(y)


# Convert categorical variables into numerical using Label Encoding
categorical_cols = ["FuelType", "Engine", "TireType", "Company","Model","Transmission","Engine", "EngineSize", "VehicleClass","Drivetrain"]  # Adjust as needed
encoder = LabelEncoder()

for col in categorical_cols:
    X[col] = encoder.fit_transform(X[col])



#training split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check for missing values
print("Missing values after preprocessing:\n", X.isnull().sum())

# Check categorical encoding
print(X.dtypes)
for col in categorical_cols:
    print(f"Unique values for {col} after encoding: {X[col].unique()}")

# Check train-test split
print(f"Training Data: {X_train.shape}, Testing Data: {X_test.shape}")
print(f"Training Labels: {y_train.shape}, Testing Labels: {y_test.shape}")

# Preview final processed data before training
print("Sample of Training Data:\n", X_train.head())
print("Sample of Training Labels:\n", y_train.head())


#step 3
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",  # Since we're predicting a continuous value (MPG)
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train the model
xgb_model.fit(X_train, y_train)

# Predict on test data
y_pred = xgb_model.predict(X_test)

# Evaluate performance (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Model RMSE: {rmse}")


