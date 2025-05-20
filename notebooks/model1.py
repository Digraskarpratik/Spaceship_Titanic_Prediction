# Import Necessary Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import logging
import pickle
import os
import streamlit as st

# Suppress Warnings
warnings.filterwarnings("ignore")

# Configure Logging
logging.basicConfig(level=logging.INFO,
                    filename="Model.log",
                    filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Import Scikit-learn Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Load Data
url = "https://raw.githubusercontent.com/Digraskarpratik/Spaceship_Titanic_Prediction/main/notebooks/train.csv"
df = pd.read_csv(url)

# Drop Irrelevant Columns
df.drop(["PassengerId", "Name", "VIP"], axis=1, inplace=True)

# Handle Missing Values
categorical_cols = ["HomePlanet", "CryoSleep", "Cabin", "Destination"]
numerical_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in numerical_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Convert Categories to Strings
for col in categorical_cols:
    df[col] = df[col].astype(str)

# Apply Label Encoding
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Target Column
df["Transported"] = df["Transported"].astype(int)

# Scale Data
scalerRS = RobustScaler()
X_train = scalerRS.fit_transform(df.drop(columns=["Transported"]))

# PCA for Dimensionality Reduction
pcs = None
for i in range(1, df.shape[1]):
    pca = PCA(n_components=i)
    pca.fit(X_train)
    evr = np.cumsum(pca.explained_variance_ratio_)
    if evr[i - 1] >= 0.90:
        pcs = i
        break

pca = PCA(n_components=pcs)
pca_data = pca.fit_transform(X_train)
pca_columns = [f"PC {j+1}" for j in range(pcs)]
pca_df = pd.DataFrame(pca_data, columns=pca_columns)
pca_df = pca_df.join(df["Transported"], how="left")

# Split Data
X = pca_df.drop(["Transported"], axis=1)
y = pca_df["Transported"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Handle Imbalanced Data
sm = SMOTE()
X_train, y_train = sm.fit_resample(X_train, y_train)

# Train Model
XGB = XGBClassifier()
XGB.fit(X_train, y_train)
y_pred_model = XGB.predict(X_test)
accuracy_score_XGB = accuracy_score(y_test, y_pred_model)
print(f"Accuracy Score for XGBoost Classifier: {round(accuracy_score_XGB * 100)}%")

# Save Model and Preprocessors
with open("xgb_model.pkl", "wb") as file:
    pickle.dump(XGB, file)

with open("scaler.pkl", "wb") as file:
    pickle.dump(scalerRS, file)

with open("pca.pkl", "wb") as file:
    pickle.dump(pca, file)
