import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title='AutoML App', layout='wide')
st.title('🔍 AutoML Interface: ML + DL Ready')

uploaded_file = st.file_uploader("📁 Upload your dataset (CSV)", type=["csv"])

algorithm = st.selectbox("🤖 Select Algorithm", [
    "KMeans (Clustering)",
    "Random Forest (Classification)",
    "Logistic Regression",
])

if uploaded_file is not None:
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
            st.dataframe(df.head())

        elif algorithm == "Random Forest (Classification)":
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
                st.warning("Target column not found. Please include a 'target' column.")

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
                st.warning("Target column not found. Please include a 'target' column.")
