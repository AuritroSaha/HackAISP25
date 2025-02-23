import pandas as pd
import tensorflow as tf
import joblib

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

# ============== VARIABLE PLACEHOLDERS (SINGLE ROW) ================
# Replace these with actual values from your front-end:
hp = 200
cylinders = 6
displacement = 3000
weight_lbs = 3500
accel_0_60 = 6.5
model_year = 2019
top_speed = 155
wheelbase_mm = 2800
track_width_mm = 1500
ground_clearance_mm = 140
frontal_area = 2.2
lift_coeff = 0.01
roofline_slope_deg = 20
name = "SomeCar V6"
fuel_type = "Gasoline"
drivetrain = "RWD"
transmission_type = "Automatic"
origin = "USA"
spoiler_wing_type = "None"
underbody_aero = "Standard"
grille_type = "Mesh"
air_vent_type = "Standard"
side_mirror_type = "Standard"

# Create the single-row dictionary
new_data = {
    "Horsepower (hp)": [hp],
    "Cylinders": [cylinders],
    "Displacement (cc)": [displacement],
    "Weight (lbs)": [weight_lbs],
    "Acceleration (0-60 mph in s)": [accel_0_60],
    "Model Year": [model_year],
    "Top Speed (mph)": [top_speed],
    "Wheelbase (mm)": [wheelbase_mm],
    "Track Width (mm)": [track_width_mm],
    "Ground Clearance (mm)": [ground_clearance_mm],
    "Frontal Area (m²)": [frontal_area],
    "Lift Coefficient": [lift_coeff],
    "Roofline Slope (degrees)": [roofline_slope_deg],
    "Name": [name],
    "Fuel Type": [fuel_type],
    "Drivetrain": [drivetrain],
    "Transmission Type": [transmission_type],
    "Origin": [origin],
    "Spoiler/Wing Type": [spoiler_wing_type],
    "Underbody Aero": [underbody_aero],
    "Grille Type": [grille_type],
    "Air Vent Type": [air_vent_type],
    "Side Mirror Type": [side_mirror_type]
}

df_new = pd.DataFrame(new_data, columns=all_columns)

# Transform
X_new = preprocessor.transform(df_new)

# Predict
prediction = model.predict(X_new)

print("Predicted Drag Coefficient:", prediction[0][0])