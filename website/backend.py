from flask import Flask, request, jsonify
import pandas as pd
import xgboost as xgb
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

########################################################################################################
# Load the pre-trained MPG model; adjust the path if necessary.
mpg_model = xgb.XGBRegressor()
mpg_model.load_model("website/MPGModel/final_xgb_model.json")

# Load the saved encoders
with open("website/MPGModel/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Define columns used during training
categorical_cols = ["FuelType", "Engine", "TireType", "Company", "Model", "Transmission", "EngineSize", "VehicleClass", "Drivetrain"]
numeric_cols = ["Year", "Weight", "Volume", "Cylinders", "WheelSize", "Horsepower", "Torque", "Price"]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        input_df = pd.DataFrame([data])

        # Convert numeric columns to float (or int) as needed
        for col in numeric_cols:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

        # Apply label encoding for categorical columns using the saved encoders
        for col in categorical_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].apply(
                    lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else -1
                )

        # Predict MPG using the XGBoost model
        mpg_pred = mpg_model.predict(input_df)[0]
        return jsonify({'mpg': "{:.1f}".format(float(mpg_pred)) })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
#################################################################################
###DRAGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG
# Load the trained drag coefficient model (e.g., "drag_coefficient_model.keras")
model = tf.keras.models.load_model("dragcoeff_dlpredict.keras")

# Load the fitted ColumnTransformer (e.g., "drag_coefficient_preprocessor.pkl")
preprocessor = joblib.load("dragcoeff_preprocessor.pkl")

# Columns for your drag coefficient model
numeric_cols = [
    "Horsepower (hp)", "Cylinders", "Displacement (cc)", "Weight (lbs)",
    "Acceleration (0-60 mph in s)", "Model Year", "Top Speed (mph)",
    "Wheelbase (mm)", "Track Width (mm)", "Ground Clearance (mm)",
    "Frontal Area (m²)", "Lift Coefficient", "Roofline Slope (degrees)"
]
categorical_cols = [
    "Name", "Fuel Type", "Drivetrain", "Transmission Type", "Origin",
    "Spoiler/Wing Type", "Underbody Aero", "Grille Type", "Air Vent Type",
    "Side Mirror Type"
]




all_columns = numeric_cols + categorical_cols


df_new = pd.DataFrame(new_data, columns=all_columns)

# Transform
X_new = preprocessor.transform(df_new)

# Predict
prediction = model.predict(X_new)

print("Predicted Drag Coefficient:", prediction[0][0])






if __name__ == '__main__':
    app.run(debug=True)
