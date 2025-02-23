import pandas as pd
import tensorflow as tf
import joblib

# Load the trained horsepower model (e.g., "horsepower_model.keras")
model = tf.keras.models.load_model("horsepower_dlpredict.keras")

# Load the fitted ColumnTransformer (e.g., "horsepower_preprocessor.pkl")
preprocessor = joblib.load("horsepower_preprocessor.pkl")

# Columns for your horsepower model
numeric_cols = ["Year", "Kilometers_Driven", "Mileage", "Engine", "Seats", "Price"]
categorical_cols = ["Name", "Location", "Fuel_Type", "Transmission", "Owner_Type"]
all_columns = numeric_cols + categorical_cols

# ============== VARIABLE PLACEHOLDERS (SINGLE ROW) ================
# Replace these with actual values from your front-end:
year = 2015
kilometers_driven = 41000
mileage = 19.6
engine = 1520
seats = 5
price = 12500
name = "Hyundai Creta 1.6 CRDi"
location = "Pune"
fuel_type = "Diesel"
transmission = "Manual"
owner_type = "First"

# Create the single-row dictionary
new_data = {
    "Year": [year],
    "Kilometers_Driven": [kilometers_driven],
    "Mileage": [mileage],
    "Engine": [engine],
    "Seats": [seats],
    "Price": [price],
    "Name": [name],
    "Location": [location],
    "Fuel_Type": [fuel_type],
    "Transmission": [transmission],
    "Owner_Type": [owner_type]
}

df_new = pd.DataFrame(new_data, columns=all_columns)

# Transform
X_new = preprocessor.transform(df_new)

# Predict
prediction = model.predict(X_new)

print("Predicted Horsepower:", (prediction[0][0])/80)