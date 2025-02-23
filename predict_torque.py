import pandas as pd
import tensorflow as tf
import joblib

# Load the trained torque model (e.g., "torque_model.keras")
model = tf.keras.models.load_model("torque_dlpredict.keras")

# Load the fitted ColumnTransformer (e.g., "torque_preprocessor.pkl")
preprocessor = joblib.load("torque_preprocessor.pkl")

# Define all columns used during training in the same order
columns = [
    "Company", "Model", "Year", "Weight", "MPG", "Volume", 
    "Cylinders", "WheelSize", "FuelType", "Transmission",
    "Engine", "EngineSize", "VehicleClass", "TireType",
    "Horsepower", "Price", "Drivetrain"
]

# ============== VARIABLE PLACEHOLDERS (SINGLE ROW) ================
# Replace these with actual values from your front-end:
company = "Honda"
model_name = "Civic"
year = 2020
weight = 2800
mpg = 32
volume = 95
cylinders = 4
wheel_size = 16
fuel_type = "Gasoline"
transmission = "Automatic"
engine = "Inline-4"
engine_size = 1.8
vehicle_class = "Sedan"
tire_type = "All-season"
horsepower = 150
price = 20000
drivetrain = "FWD"

# Create the single-row dictionary
new_data = {
    "Company": [company],
    "Model": [model_name],
    "Year": [year],
    "Weight": [weight],
    "MPG": [mpg],
    "Volume": [volume],
    "Cylinders": [cylinders],
    "WheelSize": [wheel_size],
    "FuelType": [fuel_type],
    "Transmission": [transmission],
    "Engine": [engine],
    "EngineSize": [engine_size],
    "VehicleClass": [vehicle_class],
    "TireType": [tire_type],
    "Horsepower": [horsepower],
    "Price": [price],
    "Drivetrain": [drivetrain]
}

# Convert to a DataFrame
df_new = pd.DataFrame(new_data, columns=columns)

# Transform using the preprocessor
X_new = preprocessor.transform(df_new)

# Predict
prediction = model.predict(X_new)

print("Predicted Torque:", prediction[0][0])