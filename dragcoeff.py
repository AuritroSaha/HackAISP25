import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# 1. Load the dataset
df = pd.read_csv("Datasets/detailedFactsCarsExtended.csv")

# 2. Separate features and target (assuming "Drag Coefficient" is your target)
X = df.drop("Drag Coefficient", axis=1)
y = df["Drag Coefficient"]

# 3. Identify numeric and categorical columns
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

# 4. Create a preprocessor that scales numeric features and one-hot encodes categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

# 5. Transform the feature set
X_processed = preprocessor.fit_transform(X)

# 6. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# 7. Build a TensorFlow Keras model (a simple feed-forward network)
with tf.device("/GPU:0"):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'), 
                            # kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'), 
                            # kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  # Regression output: drag coefficient
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.summary()

# 8. Train the model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train, 
                    epochs=50, 
                    batch_size=16, 
                    validation_split=0.2, 
                    callbacks=[early_stopping])

# 9. Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test, y_test)
print("Test MAE:", test_mae)

# 10. Use the model for predictions
predictions = model.predict(X_test)
print("Sample predictions:", predictions[:5])

model.save("dragcoeff_dlpredict.keras")
joblib.dump(preprocessor, "dragcoeff_preprocessor.pkl")