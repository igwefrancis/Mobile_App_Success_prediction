import streamlit as st
import pandas as pd
import joblib

# ----------------- Streamlit Page Config -----------------
st.set_page_config(
    page_title="📱 Mobile App Success Prediction & Recommendation",
    layout="wide",
    page_icon="📱"
)

# ----------------- Custom Styling -----------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 32px;
        font-weight: bold;
        color: #3B5998;
    }
    .sub-title {
        font-size: 20px;
        color: #3B5998;
    }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='main-title'>📱 Mobile App Success Prediction & Recommendation System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>An AI-powered tool to predict success and suggest similar apps</div>", unsafe_allow_html=True)

# ----------------- Load Models & Data -----------------
@st.cache_resource
def load_resources():
    prediction_model = joblib.load("model/stacked_app_success_predictor.joblib")
    knn_model = joblib.load("model/knn_model.joblib")
    knn_features_scaler = joblib.load("model/knn_features_scaler.joblib")
    prediction_features = joblib.load("model/prediction_model_features.joblib")
    knn_model_features = joblib.load("model/knn_model_features.joblib")
    successful_apps_df = joblib.load("model/successful_apps_df_processed.joblib")
    return prediction_model, knn_model, knn_features_scaler, prediction_features, knn_model_features, successful_apps_df

# Check if model files exist
try:
    stacking_model, knn_model, knn_scaler, prediction_features, knn_features, successful_apps = load_resources()
except FileNotFoundError:
    st.error("""
    **❌ Error: Model files not found!**  
    Please ensure you have generated all the required `.joblib` files and placed them inside a folder named `model`.
    """)
    st.stop()

# ----------------- Helper Functions -----------------
def get_prediction_and_recommendation(raw_input_df):
    input_features = ['Category', 'Genres', 'Content Rating', 'Type']
    pred_input_df = raw_input_df[input_features]
    
    pred_input = pd.get_dummies(pred_input_df)
    pred_input = pred_input.reindex(columns=prediction_features, fill_value=0)

    # Predict success
    prediction = stacking_model.predict(pred_input)[0]
    prediction_proba = stacking_model.predict_proba(pred_input)[0][1]

    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.success(f"✅ This app is likely to be **SUCCESSFUL!** (Confidence: {prediction_proba:.2%})")

        # Recommendation Section
        knn_features_for_transform = ['Reviews', 'Size', 'Installs', 'Price', 'Rating', 'Category', 'Genres', 'Content Rating', 'Type']
        knn_input_scaled = knn_scaler.transform(raw_input_df[knn_features_for_transform])

        distances, indices = knn_model.kneighbors(knn_input_scaled)
        recommendations = successful_apps.iloc[indices[0]][
            ["App", "Category", "Genres", "Rating", "Installs", "Price"]
        ]

        st.subheader("✨ Recommended Similar Successful Apps")
        st.dataframe(recommendations.reset_index(drop=True))

    else:
        st.error(f"❌ This app is likely to be **UNSUCCESSFUL.** (Confidence: {prediction_proba:.2%})")

# ----------------- Sidebar -----------------
with st.sidebar:
    st.image("logo1.png", use_container_width=True)  # 👈 Updated parameter
    st.markdown("## 📖 About")
    st.write(
        """
        This system uses **Machine Learning** to:
        - Predict whether an app will be successful  
        - Recommend similar successful apps  

        Built with ❤️ using **Streamlit, Scikit-learn & KNN**.
        """
    )
    st.markdown("---")
    st.write("👨‍💻 Developed by: *Igwe Francis N*")

# ----------------- Main Section -----------------
st.header("🔍 Select an App for Prediction & Recommendation")

app_list = successful_apps["App"].unique()
selected_app = st.selectbox("Choose an app", app_list)

if st.button("🚀 Predict & Recommend"):
    app_data_df = successful_apps[successful_apps["App"] == selected_app]
    st.write(f"Analyzing **{selected_app}**...")
    get_prediction_and_recommendation(app_data_df)
