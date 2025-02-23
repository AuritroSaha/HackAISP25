import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ----------------------------------------------------------------------
# 1) LOAD THE DATASET
# ----------------------------------------------------------------------
df = pd.read_csv("Datasets/dataGen.csv")

# The CSV columns are (for reference):
# Company,Model,Year,Weight,MPG,Volume,Cylinders,WheelSize,FuelType,
# Transmission,Engine,EngineSize,VehicleClass,TireType,Horsepower,
# Torque,Price,Drivetrain

# ----------------------------------------------------------------------
# 2) SEPARATE FEATURES (X) AND TARGET (y)
# ----------------------------------------------------------------------
# We want to predict Torque, so drop it from the features
X = df.drop("Torque", axis=1)
y = df["Torque"]

# ----------------------------------------------------------------------
# 3) DECIDE WHICH COLUMNS ARE NUMERIC VS. CATEGORICAL
# ----------------------------------------------------------------------
# Numeric columns (continuous or integer)
numeric_cols = [
    "Year", 
    "Weight", 
    "MPG", 
    "Volume", 
    "Cylinders",
    "WheelSize",
    "EngineSize", 
    "Horsepower",
    "Price"
]

# Categorical columns
categorical_cols = [
    "Company", 
    "Model", 
    "FuelType", 
    "Transmission", 
    "Engine", 
    "VehicleClass", 
    "TireType", 
    "Drivetrain"
]

# If any of these columns are missing in your DataFrame, remove them:
for col in numeric_cols.copy():
    if col not in X.columns:
        numeric_cols.remove(col)

for col in categorical_cols.copy():
    if col not in X.columns:
        categorical_cols.remove(col)

# ----------------------------------------------------------------------
# 4) PREPROCESSING PIPELINE:
#    - StandardScaler for numeric columns
#    - OneHotEncoder for categorical columns
# ----------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# ----------------------------------------------------------------------
# 5) SPLIT INTO TRAIN/TEST SETS AND FIT THE PREPROCESSOR
# ----------------------------------------------------------------------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor.fit(X_train_raw)

X_train = preprocessor.transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

# ----------------------------------------------------------------------
# 6) BUILD & COMPILE A TENSORFLOW MODEL (ON GPU IF AVAILABLE)
# ----------------------------------------------------------------------
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# If you have a GPU, this ensures the model runs on /GPU:0
# You can omit the `with tf.device("/GPU:0"):` block for automatic GPU usage
with tf.device("/GPU:0"):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  # single numeric output for Torque
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

model.summary()

# ----------------------------------------------------------------------
# 7) TRAIN THE MODEL WITH EARLY STOPPING
# ----------------------------------------------------------------------
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, 
    y_train,
    validation_split=0.2, 
    epochs=50, 
    batch_size=32, 
    callbacks=[early_stopping]
)

# ----------------------------------------------------------------------
# 8) EVALUATE ON THE TEST SET
# ----------------------------------------------------------------------
test_loss, test_mae = model.evaluate(X_test, y_test)
print("Test MSE:", test_loss)
print("Test MAE:", test_mae)

# ----------------------------------------------------------------------
# 9) PREDICT ON THE TEST SET
# ----------------------------------------------------------------------
predictions = model.predict(X_test)
print("First 5 predictions:", predictions[:5].ravel())
print("First 5 actual:", y_test.iloc[:5].values)

# ----------------------------------------------------------------------
# 10) (OPTIONAL) SAVE THE MODEL
# ----------------------------------------------------------------------
model.save("torque_dlpredict.keras")
joblib.dump(preprocessor, "torque_preprocessor.pkl")