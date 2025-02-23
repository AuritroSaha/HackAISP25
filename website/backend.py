from flask import Flask, request, jsonify
import pandas as pd
import xgboost as xgb
import tensorflow as tf

app = Flask(__name__)

# Load the MPG model (from your mpgmodel.py)
mpg_model = xgb.XGBRegressor()
mpg_model.load_model("final_xgb_model.json")

# Load the drag coefficient model (your keras model)
drag_model = tf.keras.models.load_model("dragcoeff_dlpredict.keras")

# If you need to encode categorical features as in your training script,
# you’ll need to load the same encoders (for example, from saved pickle files).

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()

    # Create a DataFrame from the input (ensure your keys match your training features)
    input_df = pd.DataFrame([data])
    
    # Perform any necessary preprocessing, for example:
    # for col in categorical_cols:
    #     input_df[col] = encoders[col].transform(input_df[col])
    
    # Predict MPG using your xgboost model
    mpg_pred = mpg_model.predict(input_df)[0]
    
    # Predict drag coefficient using your keras model
    # (Assuming the keras model accepts the same input structure)
    drag_pred = drag_model.predict(input_df)[0][0]
    
    # Return predictions as JSON
    return jsonify({
        'mpg': mpg_pred,
        'drag_coefficient': drag_pred
    })

if __name__ == '__main__':
    app.run(debug=True)
