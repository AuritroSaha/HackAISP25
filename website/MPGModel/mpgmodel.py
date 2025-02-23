import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load dataset (replace with actual file path)
df = pd.read_csv("Datasets/dataGen.csv")

# Define the target variable (MPG)
target_column = "MPG"  # Change this if MPG has a different column name
X = df.drop(columns=[target_column])  # Features
y = df[target_column]  # Target

# Define categorical columns
categorical_cols = ["FuelType", "Engine", "TireType", "Company", "Model", "Transmission", "EngineSize", "VehicleClass", "Drivetrain"]

# Dictionary to store encoders for later use
encoders = {}

# Convert categorical variables into numerical using Label Encoding
for col in categorical_cols:
    if col in X.columns:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])
        encoders[col] = encoder  # Store encoder for later use

# Training split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the final model using best hyperparameters
final_xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=300,
    learning_rate=0.3,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

final_xgb_model.fit(X_train, y_train)
print("✅ Final Model Trained Successfully!")

# Evaluate model performance
y_pred = final_xgb_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"📊 Final Model Performance:")
print(f"✅ RMSE: {rmse}")
print(f"✅ R² Score: {r2}")

# Compute Training RMSE
y_train_pred = final_xgb_model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n📉 Overfitting Check:")
print(f"Training RMSE: {train_rmse}")
print(f"Testing RMSE: {test_rmse}")

if train_rmse < test_rmse * 0.9:
    print("⚠️ Warning: The model may be overfitting.")
elif train_rmse > test_rmse * 1.2:
    print("⚠️ Warning: The model may be underfitting.")
else:
    print("✅ Model is generalizing well.")

# Save the trained model
final_xgb_model.save_model("final_xgb_model.json")
print("✅ Model saved successfully!")

import pickle
# Save the label encoders to a pickle file
with open("website/MPGModel/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
print("✅ Encoders saved successfully!")




# Load the saved model
loaded_model = xgb.XGBRegressor()
loaded_model.load_model("final_xgb_model.json")

# Example new data (replace with real values)
new_car = pd.DataFrame({
    "Company": ["BMW"],
    "Model": ["Pilot"],
    "Year": [2014],
    "Weight": [4500],
    "Volume": [145],
    "Cylinders": [8],
    "WheelSize": [18],
    "FuelType": ["Gasoline"],
    "Transmission": ["Automatic"],
    "Engine": ["V6"],
    "EngineSize": [3.5],
    "VehicleClass": ["SUV"],
    "TireType": ["All-Season"],
    "Horsepower": [284],
    "Torque": [266],
    "Price": [34560],
    "Drivetrain": ["FWD"]
})

# Ensure new_car has the same columns as training data
for col in categorical_cols:
    if col in new_car.columns:
        if col in encoders:  # Ensure encoder exists
            new_car[col] = new_car[col].apply(lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else -1)

# Predict MPG for new data
predicted_mpg = final_xgb_model.predict(new_car)
print(f"🚗 Predicted MPG: {predicted_mpg[0]}")
