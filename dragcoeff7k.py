import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ----------------------------------------------------------------------
# 1) LOAD THE DATASET
#    Adjust the CSV path as needed
# ----------------------------------------------------------------------
df = pd.read_csv("Datasets/used_cars_data.csv")

# Columns in the data (for reference):
# S.No., Name, Location, Year, Kilometers_Driven, Fuel_Type,
# Transmission, Owner_Type, Mileage, Engine, Power, Seats, New_Price, Price

# ----------------------------------------------------------------------
# 2) DROP COLUMNS WE DON'T NEED
#    We'll drop S.No. (just an index) and New_Price (might be messy)
# ----------------------------------------------------------------------
df.drop(["S.No.", "New_Price"], axis=1, errors="ignore", inplace=True)

# ----------------------------------------------------------------------
# 3) DEFINE A GENERAL CLEANING FUNCTION FOR NUMERIC COLUMNS
# ----------------------------------------------------------------------
def clean_numeric_column(series, substrings_to_remove=None):
    """
    1. Convert to string.
    2. Remove given substrings (e.g., 'kmpl', 'bhp', 'CC', 'Lakh').
    3. Remove all non-digit and non-dot characters.
    4. Convert to float (NaN if invalid).
    """
    if substrings_to_remove is None:
        substrings_to_remove = []

    # Convert to string
    series = series.astype(str).copy()

    # Replace known placeholders like 'null'
    series.replace("null", "", inplace=True)  # e.g. if your CSV has literal "null"

    # Remove the specified units/text substrings
    for sub in substrings_to_remove:
        series = series.str.replace(sub, "", regex=False)

    # Remove any leftover characters that are not digits or decimal points
    series = series.str.replace(r"[^0-9.]", "", regex=True)

    # Convert to float, invalid strings become NaN
    return pd.to_numeric(series, errors="coerce")


# ----------------------------------------------------------------------
# 4) CLEAN EACH NUMERIC COLUMN
#    Adjust substrings_to_remove as needed for your data
# ----------------------------------------------------------------------

# Mileage might have "kmpl" or "km/kg"
df["Mileage"] = clean_numeric_column(
    df["Mileage"], 
    substrings_to_remove=["kmpl", "km/kg"]
)

# Engine might have "CC"
df["Engine"] = clean_numeric_column(
    df["Engine"], 
    substrings_to_remove=["CC"]
)

# Power might have "bhp"
df["Power"] = clean_numeric_column(
    df["Power"], 
    substrings_to_remove=["bhp"]
)

# Price might have "Lakh", "Cr", etc.
df["Price"] = clean_numeric_column(
    df["Price"], 
    substrings_to_remove=["Lakh", "Cr"]
)

# Seats typically is just an integer, but we can clean it anyway
df["Seats"] = clean_numeric_column(df["Seats"])

# Year is usually an integer, but let's ensure it's numeric
df["Year"] = clean_numeric_column(df["Year"])

# Kilometers_Driven sometimes might have extra text, so let's clean it too
df["Kilometers_Driven"] = clean_numeric_column(df["Kilometers_Driven"])

# ----------------------------------------------------------------------
# 5) DROP ROWS STILL MISSING CRUCIAL COLUMNS
#    Especially the target 'Power'
# ----------------------------------------------------------------------
df.dropna(subset=["Power"], inplace=True)

# We also need good numeric data for modeling in columns like Year, Mileage, etc.
numeric_columns = ["Year", "Kilometers_Driven", "Mileage", "Engine", "Seats", "Price"]
df.dropna(subset=numeric_columns, inplace=True)

# ----------------------------------------------------------------------
# 6) SEPARATE FEATURES (X) AND TARGET (y)
#    We'll predict Power
# ----------------------------------------------------------------------
X = df.drop("Power", axis=1)
y = df["Power"]

# ----------------------------------------------------------------------
# 7) IDENTIFY NUMERIC AND CATEGORICAL COLUMNS
# ----------------------------------------------------------------------
numeric_cols = ["Year", "Kilometers_Driven", "Mileage", "Engine", "Seats", "Price"]
categorical_cols = ["Name", "Location", "Fuel_Type", "Transmission", "Owner_Type"]

# Some columns might be missing, so let's remove them from the lists if they don't exist:
for col in numeric_cols.copy():
    if col not in X.columns:
        numeric_cols.remove(col)

for col in categorical_cols.copy():
    if col not in X.columns:
        categorical_cols.remove(col)

# ----------------------------------------------------------------------
# 8) BUILD A PREPROCESSING PIPELINE
#    Scale numeric columns, one-hot encode categorical columns
# ----------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# ----------------------------------------------------------------------
# 9) SPLIT THE DATA & FIT THE PREPROCESSOR
# ----------------------------------------------------------------------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor.fit(X_train_raw)
X_train = preprocessor.transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# ----------------------------------------------------------------------
# 10) BUILD & COMPILE TENSORFLOW MODEL
# ----------------------------------------------------------------------
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

with tf.device("/GPU:0"):  # or omit to let TF auto-detect GPU
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  # single numeric output (Power)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.summary()

# ----------------------------------------------------------------------
# 11) TRAIN WITH EARLY STOPPING
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
# 12) EVALUATE ON THE TEST SET
# ----------------------------------------------------------------------
test_loss, test_mae = model.evaluate(X_test, y_test)
print("Test MSE:", test_loss)
print("Test MAE:", test_mae)

# ----------------------------------------------------------------------
# 13) EXAMPLE PREDICTIONS
# ----------------------------------------------------------------------
predictions = model.predict(X_test)
print("First 5 predictions:", predictions[:5].ravel())
print("First 5 actual:", y_test.iloc[:5].values)