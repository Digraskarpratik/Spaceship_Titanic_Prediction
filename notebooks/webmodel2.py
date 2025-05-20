import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os


st.set_page_config(page_title="Spaceship Titanic Prediction", layout="centered")
st.title("Spaceship Titanic Prediction App")

# Load saved model & preprocessors
base_path = os.path.dirname(__file__)  # Dynamic path for Streamlit deployment

try:
    with open(os.path.join(base_path, "xgb_model.pkl"), "rb") as file:
        model = pickle.load(file)
    with open(os.path.join(base_path, "scaler.pkl"), "rb") as file:
        scaler = pickle.load(file)
    with open(os.path.join(base_path, "pca.pkl"), "rb") as file:
        pca = pickle.load(file)
except FileNotFoundError:
    st.error("Model files missing! Ensure you have trained and saved the models.")

# Input Form
st.header("Enter Passenger Details")
homeplanet = st.selectbox("HomePlanet", ["Earth", "Europa", "Mars"])
cryosleep = st.selectbox("CryoSleep", ["True", "False"])
cabin = st.selectbox("Cabin Deck", ["A", "B", "C", "D", "E"])  # Simplified
destination = st.selectbox("Destination", ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"])
age = st.slider("Age", 0, 100, 30)
room_service = st.number_input("RoomService", min_value=0)
food_court = st.number_input("FoodCourt", min_value=0)
shopping_mall = st.number_input("ShoppingMall", min_value=0)
spa = st.number_input("Spa", min_value=0)
vr_deck = st.number_input("VRDeck", min_value=0)

# Label Encoding based on training
homeplanet_map = {"Earth": 0, "Europa": 1, "Mars": 2}
cryosleep_map = {"False": 0, "True": 1}
cabin_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
destination_map = {"TRAPPIST-1e": 2, "55 Cancri e": 0, "PSO J318.5-22": 1}

input_dict = {
    "HomePlanet": homeplanet_map[homeplanet],
    "CryoSleep": cryosleep_map[cryosleep],
    "Cabin": cabin_map[cabin],
    "Destination": destination_map[destination],
    "Age": age,
    "RoomService": room_service,
    "FoodCourt": food_court,
    "ShoppingMall": shopping_mall,
    "Spa": spa,
    "VRDeck": vr_deck
}

input_df = pd.DataFrame([input_dict])

# Ensure input_df matches scaler training
input_df = input_df[scaler.feature_names_in_]

# Apply Preprocessing
scaled = scaler.transform(input_df)
transformed = pca.transform(scaled)

# Prediction
if st.button("Predict"):
    prediction = model.predict(transformed)
    result = "Transported" if prediction[0] == 1 else "Not Transported"
    st.success(f"Prediction: {result}")