import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title='🧠 AutoML Interface', layout='wide')
st.title("🔍 Smart AutoML Interface")

# === Step 1: Select AI Category ===
category = st.selectbox("📂 Select AI Category", ["-- Select --", "ML (Machine Learning)", "DL (Deep Learning)", "CV (Computer Vision)"])

# === Step 2: Show Relevant Algorithms ===
algo_options = []

if category == "ML (Machine Learning)":
    algo_options = ["KMeans (Clustering)", "Random Forest", "Logistic Regression"]
elif category == "DL (Deep Learning)":
    algo_options = ["Simple Neural Network (CSV)"]
elif category == "CV (Computer Vision)":
    algo_options = ["Image Classification (CNN)"]

algorithm = st.selectbox("🤖 Select Algorithm", ["-- Select --"] + algo_options)

# === Step 3: Upload Dataset ===
st.subheader("📁 Upload Your Dataset")

if category == "CV (Computer Vision)":
    uploaded_file = st.file_uploader("Upload a ZIP of image folder (CV only)", type=["zip"])
else:
    uploaded_file = st.file_uploader("Upload a CSV file (ML/DL)", type=["csv"])

# === Step 4: Validate + Run ===
if uploaded_file is not None and algorithm != "-- Select --":

    # Error if wrong file type
    if category == "CV (Computer Vision)" and not uploaded_file.name.endswith(".zip"):
        st.error("❌ This algorithm requires image ZIP dataset. Please upload a valid image zip file.")
    elif category in ["ML (Machine Learning)", "DL (Deep Learning)"] and not uploaded_file.name.endswith(".csv"):
        st.error("❌ This algorithm requires a CSV dataset. Please upload a valid .csv file.")

    # === CSV Dataset ===
    elif uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head())

        X = df.select_dtypes(include=np.number)
        y = df['target'] if 'target' in df.columns else None

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if st.button("🚀 Run Model"):
            st.subheader("📊 Result")
            if algorithm == "KMeans (Clustering)":
                k = st.slider("Choose number of clusters", 2, 10, 3)
                model = KMeans(n_clusters=k)
                labels = model.fit_predict(X_scaled)
                df['Cluster'] = labels
                fig, ax = plt.subplots()
                ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='Set2')
                ax.set_title("KMeans Clusters")
                st.pyplot(fig)

            elif algorithm == "Random Forest":
                if y is not None:
                    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
                    model = RandomForestClassifier()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    st.text("Confusion Matrix:")
                    st.text(confusion_matrix(y_test, y_pred))
                    st.text("Classification Report:")
                    st.text(classification_report(y_test, y_pred))
                else:
                    st.warning("Please add a 'target' column to use this algorithm.")

            elif algorithm == "Logistic Regression":
                if y is not None:
                    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
                    model = LogisticRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    st.text("Confusion Matrix:")
                    st.text(confusion_matrix(y_test, y_pred))
                    st.text("Classification Report:")
                    st.text(classification_report(y_test, y_pred))
                else:
                    st.warning("Please add a 'target' column to use this algorithm.")

            elif algorithm == "Simple Neural Network (CSV)":
                st.warning("⚠️ DL model support coming soon! Currently under development.")

    # === Image ZIP Dataset ===
    elif uploaded_file.name.endswith(".zip") and category == "CV (Computer Vision)":
        st.warning("⚠️ CV image model support (CNN) is coming soon! Not yet implemented.")
