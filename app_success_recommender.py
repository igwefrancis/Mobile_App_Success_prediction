import streamlit as st
import pandas as pd
import joblib

# ----------------- Streamlit Page Config -----------------
st.set_page_config(page_title="📱 Mobile App Success Prediction & Recommendation", layout="wide")
st.title("📱 Mobile App Success Prediction & Recommendation System")

# ----------------- Load Models & Data -----------------
@st.cache_resource
def load_resources():
    """
    Loads all the necessary models and dataframes from the 'model' directory.
    Uses st.cache_resource to ensure these heavy objects are loaded only once.
    """
    prediction_model = joblib.load("model/stacked_app_success_predictor.joblib")
    knn_model = joblib.load("model/knn_model.joblib")
    knn_features_scaler = joblib.load("model/knn_features_scaler.joblib")
    prediction_features = joblib.load("model/prediction_model_features.joblib")
    knn_model_features = joblib.load("model/knn_model_features.joblib")
    # This dataframe contains the original categorical columns for display and input
    successful_apps_df = joblib.load("model/successful_apps_df_processed.joblib")
    return prediction_model, knn_model, knn_features_scaler, prediction_features, knn_model_features, successful_apps_df

# Check if model files exist
try:
    stacking_model, knn_model, knn_scaler, prediction_features, knn_features, successful_apps = load_resources()
except FileNotFoundError:
    st.error("""
    **Error: Model files not found!** Please ensure you have generated all the required `.joblib` files and placed them in a folder named `model` in the same directory as this script.
    """)
    st.stop()

# ----------------- Helper Functions -----------------
def get_prediction_and_recommendation(raw_input_df):
    """
    Predicts app success and provides recommendations based on raw input.
    """
    # 1. Prepare data for prediction
    input_features = ['Category', 'Genres', 'Content Rating', 'Type']
    pred_input_df = raw_input_df[input_features]
    
    pred_input = pd.get_dummies(pred_input_df)
    pred_input = pred_input.reindex(columns=prediction_features, fill_value=0)

    # 2. Predict success
    prediction = stacking_model.predict(pred_input)[0]
    prediction_proba = stacking_model.predict_proba(pred_input)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"✅ This app is likely to be **SUCCESSFUL!** (Confidence: {prediction_proba:.2%})")

        # 3. Prepare data for recommendation
        # IMPORTANT: Use the original feature names for the scaler
        # The scaler itself handles the transformation to the correct features
        knn_features_for_transform = ['Reviews', 'Size', 'Installs', 'Price', 'Rating', 'Category', 'Genres', 'Content Rating', 'Type']
        knn_input_scaled = knn_scaler.transform(raw_input_df[knn_features_for_transform])

        # 4. Find similar apps
        distances, indices = knn_model.kneighbors(knn_input_scaled)
        recommendations = successful_apps.iloc[indices[0]][
            ["App", "Category", "Genres", "Rating", "Installs", "Price"]
        ]

        st.subheader("Recommended Similar Successful Apps")
        st.dataframe(recommendations.reset_index(drop=True))

    else:
        st.error(f"❌ This app is likely to be **UNSUCCESSFUL.** (Confidence: {prediction_proba:.2%})")

# ----------------- Select App Section -----------------
st.header("Select an App from the List")

# Use the 'successful_apps' dataframe to populate the dropdown
app_list = successful_apps["App"].unique()
selected_app = st.selectbox("Choose an app", app_list)

# Use a button to trigger the prediction and recommendation for the selected app
if st.button("Predict & Recommend for Selected App"):
    # Find the row for the selected app
    app_data_df = successful_apps[successful_apps["App"] == selected_app]
    
    st.write(f"Predicting success for **{selected_app}**...")
    get_prediction_and_recommendation(app_data_df)