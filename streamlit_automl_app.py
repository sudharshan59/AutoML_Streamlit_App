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
import tensorflow as tf

st.set_page_config(page_title='🧠 AutoML System', layout='wide')
st.title("🤖 AutoML Web App (ML + DL + CV)")

# === Step 1: Select Category ===
category = st.radio("📂 Choose AI Category", ["ML", "DL", "CV"])

# === Step 2: Upload Dataset ===
if category == "CV":
    uploaded = st.file_uploader("Upload a ZIP of images (CV only)", type=["zip"])
else:
    uploaded = st.file_uploader("Upload a CSV dataset (ML/DL)", type=["csv"])

# === Step 3: Handle Dataset
if uploaded:
    if category == "CV" and not uploaded.name.endswith(".zip"):
        st.error("❌ Please upload a valid .ZIP file for CV models.")
    elif category in ["ML", "DL"] and not uploaded.name.endswith(".csv"):
        st.error("❌ Please upload a valid .CSV file for ML/DL models.")
    else:
        if category in ["ML", "DL"]:
            df = pd.read_csv(uploaded)
            st.subheader("📊 Dataset Preview")
            st.dataframe(df.head())

            X = df.select_dtypes(include=np.number)
            y = df['target'] if 'target' in df.columns else None

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            if st.button("🚀 Run All Models"):
                st.success(f"Running all models in {category}...")

                # ==== ML MODELS ====
                if category == "ML":
                    st.subheader("🧠 ML: KMeans (Clustering)")
                    kmeans = KMeans(n_clusters=3)
                    labels = kmeans.fit_predict(X_scaled)
                    fig, ax = plt.subplots()
                    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='Set2')
                    ax.set_title("KMeans Clusters")
                    st.pyplot(fig)

                    if y is not None:
                        # Random Forest
                        st.subheader("🌲 Random Forest")
                        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
                        rf = RandomForestClassifier()
                        rf.fit(X_train, y_train)
                        y_pred = rf.predict(X_test)
                        st.text("Confusion Matrix:")
                        st.text(confusion_matrix(y_test, y_pred))
                        st.text("Classification Report:")
                        st.text(classification_report(y_test, y_pred))

                        # Logistic Regression
                        st.subheader("📈 Logistic Regression")
                        model = LogisticRegression()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        st.text("Confusion Matrix:")
                        st.text(confusion_matrix(y_test, y_pred))
                        st.text("Classification Report:")
                        st.text(classification_report(y_test, y_pred))
                    else:
                        st.warning("⚠️ 'target' column not found. Classification models skipped.")

                # ==== DL MODELS ====
                elif category == "DL":
                    if y is not None:
                        st.subheader("🔮 Simple Neural Network")
                        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
                        model = tf.keras.Sequential([
                            tf.keras.layers.Dense(64, activation='relu'),
                            tf.keras.layers.Dense(32, activation='relu'),
                            tf.keras.layers.Dense(1, activation='sigmoid')
                        ])
                        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                        model.fit(X_train, y_train, epochs=10, verbose=0)
                        loss, acc = model.evaluate(X_test, y_test)
                        st.success(f"✅ Neural Network Accuracy: {acc:.2%}")
                    else:
                        st.warning("⚠️ 'target' column not found. Neural network training skipped.")

        # ==== CV PLACEHOLDER ====
        elif category == "CV":
            st.warning("📦 CV model support coming soon (CNN on image ZIP).")

# === Step 4: Built-in AI Assistant ===
st.markdown("---")
st.subheader("💬 Ask the AutoML Assistant")

query = st.text_input("🧠 Ask about dataset, models, or errors...")

if query:
    query = query.lower()
    if "target" in query:
        answer = "A 'target' column is required for classification models like Random Forest or DL."
    elif "kmeans" in query:
        answer = "KMeans is unsupervised and does not need a target column."
    elif "csv" in query:
        answer = "CSV files are used for ML and DL models. Make sure features are numeric."
    elif "zip" in query or "image" in query:
        answer = "ZIP files should contain images organized in folders by class (for CV models)."
    else:
        answer = "Try asking about model types, dataset formats, or error messages."

    st.info(f"💡 {answer}")
