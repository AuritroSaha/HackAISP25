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
# ===================== Drag Coefficient Model Setup =====================
# Define columns for your drag coefficient model
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
# Define columns for your drag coefficient model
numeric_cols = [
    "Horsepower (hp)", "Cylinders", "Displacement (cc)", "Weight (lbs)",
    "Acceleration (0-60 mph in s)", "Model Year", "Top Speed (mph)",
    "Wheelbase (mm)", "Track Width (mm)", "Ground Clearance (mm)",
    "Frontal Area (m²)", "Lift Coefficient", "Roofline Slope (degrees)"
]
categorical_cols = [
    "Name", "Fuel Type", "Drivetrain", "Transmission Type", "Origin",
    "Spoiler/Wing Type", "Underbody Aero", "Grille Type", "Air Vent Type", "Side Mirror Type"
]
all_columns = numeric_cols + categorical_cols

# Create aliases matching your original variable names
drag_numeric_cols = numeric_cols
drag_all_columns = all_columns

# Load the trained Keras model and preprocessor
drag_model = tf.keras.models.load_model("dragcoeff_dlpredict.keras")
drag_preprocessor = joblib.load("dragcoeff_preprocessor.pkl")

@app.route('/predict_drag', methods=['POST'])
def predict_drag():
    try:
        # Get JSON data for drag coefficient prediction
        data = request.get_json()
        
        # Rename incoming keys to match expected column names
        # This mapping converts keys sent from the front end to the names expected by the model.
        rename_map = {
            "ModelYear": "Model Year",
            "FuelType": "Fuel Type",
            "TransmissionType": "Transmission Type",
            "DisplacementCC": "Displacement (cc)",
            "WeightLbs": "Weight (lbs)",
            "SpoilerWingType": "Spoiler/Wing Type",
            "FrontalAreaM2": "Frontal Area (m²)",
            "RooflineSlopeDegrees": "Roofline Slope (degrees)",
            "Horsepower": "Horsepower (hp)",
            "AirVentType": "Air Vent Type",
            "SideMirrorType": "Side Mirror Type"
        }
        for key, new_key in rename_map.items():
            if key in data:
                data[new_key] = data.pop(key)
        
        # Create DataFrame and reindex to ensure all required columns are present.
        input_df = pd.DataFrame([data])
        input_df = input_df.reindex(columns=drag_all_columns)
        
        # Convert numeric columns to numeric type and fill missing values with 0.
        for col in drag_numeric_cols:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0)
        
        # For categorical columns, fill missing values with empty strings.
        for col in categorical_cols:
            input_df[col] = input_df[col].fillna("")
        
        # Preprocess the data using the fitted ColumnTransformer
        X_new = drag_preprocessor.transform(input_df)

        # Predict the drag coefficient using the loaded Keras model
        drag_pred = drag_model.predict(X_new)
        # Squeeze to extract a scalar value (handles both (1,1) and (1,) shapes)
        drag_value = float(drag_pred.squeeze())
        
        return jsonify({'drag_coefficient': drag_value})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

#############################################
# Run the Flask Application
#############################################
if __name__ == '__main__':
    app.run(debug=True)
