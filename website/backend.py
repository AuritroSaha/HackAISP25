from flask import Flask, request, jsonify
import pandas as pd
import xgboost as xgb
from flask_cors import CORS
import pickle
import tensorflow as tf
import joblib

app = Flask(__name__)
CORS(app)

#############################################
# MPG Prediction Endpoint
#############################################
# Load the pre-trained MPG model; adjust the path if necessary.
mpg_model = xgb.XGBRegressor()
mpg_model.load_model("website/MPGModel/final_xgb_model.json")

# Load the saved encoders for MPG
with open("website/MPGModel/encoders.pkl", "rb") as f:
    mpg_encoders = pickle.load(f)

# Define columns used during MPG training
mpg_categorical_cols = ["FuelType", "Engine", "TireType", "Company", "Model", "Transmission", "EngineSize", "VehicleClass", "Drivetrain"]
mpg_numeric_cols = ["Year", "Weight", "Volume", "Cylinders", "WheelSize", "Horsepower", "Torque", "Price"]

@app.route('/predict', methods=['POST'])
def predict_mpg():
    try:
        # Get JSON data from the request
        data = request.get_json()
        input_df = pd.DataFrame([data])

        # Convert numeric columns to float (or int) as needed
        for col in mpg_numeric_cols:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

        # Apply label encoding for categorical columns using the saved encoders
        for col in mpg_categorical_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].apply(
                    lambda x: mpg_encoders[col].transform([x])[0] if x in mpg_encoders[col].classes_ else -1
                )

        # Predict MPG using the XGBoost model
        mpg_pred = mpg_model.predict(input_df)[0]
        return jsonify({'mpg': "{:.1f}".format(float(mpg_pred))})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

#############################################
# Drag Coefficient Prediction Endpoint
#############################################
# Load the trained drag coefficient model (e.g., "dragcoeff_dlpredict.keras")
drag_model = tf.keras.models.load_model("dragcoeff_dlpredict.keras")

# Load the fitted ColumnTransformer (e.g., "dragcoeff_preprocessor.pkl")
drag_preprocessor = joblib.load("dragcoeff_preprocessor.pkl")

# Define columns for your drag coefficient model
drag_numeric_cols = [
    "Horsepower (hp)", "Cylinders", "Displacement (cc)", "Weight (lbs)",
    "Acceleration (0-60 mph in s)", "Model Year", "Top Speed (mph)",
    "Wheelbase (mm)", "Track Width (mm)", "Ground Clearance (mm)",
    "Frontal Area (m²)", "Lift Coefficient", "Roofline Slope (degrees)"
]
drag_categorical_cols = [
    "Name", "Fuel Type", "Drivetrain", "Transmission Type", "Origin",
    "Spoiler/Wing Type", "Underbody Aero", "Grille Type", "Air Vent Type", "Side Mirror Type"
]

# Combine all required columns
drag_all_columns = drag_numeric_cols + drag_categorical_cols

@app.route('/predict_drag', methods=['POST'])
def predict_drag():
    try:
        # Get JSON data for drag coefficient prediction
        data = request.get_json()
        input_df = pd.DataFrame([data])
        
        # Ensure that the DataFrame has the required columns in the correct order
        input_df = input_df[drag_all_columns]

        # Convert numeric columns to numeric types
        for col in drag_numeric_cols:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors="coerce")
        
        # Preprocess the data using the fitted ColumnTransformer
        X_new = drag_preprocessor.transform(input_df)

        # Predict the drag coefficient using the loaded Keras model
        drag_pred = drag_model.predict(X_new)
        # Assuming the model returns a 2D array (n, 1), extract the value
        drag_value = float(drag_pred[0][0])
        return jsonify({'drag_coefficient': drag_value})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

#############################################
# Run the Flask Application
#############################################
if __name__ == '__main__':
    app.run(debug=True)
