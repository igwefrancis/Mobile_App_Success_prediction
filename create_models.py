import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
import os

# Create the model directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# ----------------- 1. Data Loading and Cleaning -----------------
print("Step 1: Loading and cleaning data...")
try:
    df = pd.read_csv('googleplaystore.csv')
except FileNotFoundError:
    print("Error: googleplaystore.csv not found. Please ensure the CSV file is in the same directory as this script.")
    exit()

# Function to safely clean and convert values
def clean_value(s, col_name):
    if not isinstance(s, str):
        return s
    
    s = s.replace(',', '').replace('+', '').strip()
    
    if 'M' in s and col_name == 'Size':
        return float(s.replace('M', ''))
    elif 'k' in s and col_name == 'Size':
        return float(s.replace('k', '')) / 1000
    elif 'Varies with device' in s:
        return np.nan
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan

# Define columns to clean and their target types
columns_to_clean = {
    'Reviews': 'int',
    'Size': 'float',
    'Installs': 'int',
    'Price': 'float',
    'Rating': 'float'
}

# Apply cleaning function to relevant columns
for col, dtype in columns_to_clean.items():
    if col in df.columns:
        df[col] = df[col].apply(lambda x: clean_value(x, col))

# Drop any rows with remaining NaN values after cleaning
df.dropna(how='any', inplace=True)

# ----------------- 2. Feature Engineering & Preprocessing -----------------
print("Step 2: Engineering features...")
df['App Success'] = (df['Installs'] >= 1000000).astype(int)

# Separate features (X) and target (y)
X = df.drop('App Success', axis=1)
y = df['App Success']

# Identify features to be used in the models
categorical_features = ['Category', 'Genres', 'Content Rating', 'Type']
numerical_features = ['Reviews', 'Size', 'Installs', 'Price', 'Rating']

# ----------------- 3. Prediction Model Training & Saving -----------------
print("Step 3: Training and saving the prediction model...")

X_pred = X[categorical_features]
y_pred = y

one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
X_pred_encoded = one_hot_encoder.fit_transform(X_pred)
prediction_features = one_hot_encoder.get_feature_names_out(categorical_features)

estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('mlp', MLPClassifier(random_state=42))
]
stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5)
stacking_model.fit(X_pred_encoded, y_pred)

joblib.dump(stacking_model, 'model/stacked_app_success_predictor.joblib')
joblib.dump(prediction_features, 'model/prediction_model_features.joblib')
print("✅ Prediction model and features saved successfully!")

# ----------------- 4. Recommender Model Training & Saving -----------------
print("Step 4: Training and saving the recommender model...")
successful_apps_df = df[df['App Success'] == 1].copy()

knn_features = ['Category', 'Genres', 'Content Rating', 'Type', 'Reviews', 'Size', 'Installs', 'Price', 'Rating']
X_knn = successful_apps_df[knn_features]

preprocessor_knn = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Reviews', 'Size', 'Installs', 'Price', 'Rating']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Category', 'Genres', 'Content Rating', 'Type'])
    ])

knn_pipeline = Pipeline(steps=[('preprocessor', preprocessor_knn), ('knn', NearestNeighbors(n_neighbors=6, metric='cosine'))])
knn_pipeline.fit(X_knn)

joblib.dump(knn_pipeline.named_steps['knn'], 'model/knn_model.joblib')
joblib.dump(knn_pipeline.named_steps['preprocessor'], 'model/knn_features_scaler.joblib')
joblib.dump(preprocessor_knn.get_feature_names_out(), 'model/knn_model_features.joblib')
joblib.dump(successful_apps_df, 'model/successful_apps_df_processed.joblib')
print("✅ Recommender model and processed dataframe saved successfully!")

print("\nAll `.joblib` files have been created. You can now run your Streamlit app.")
