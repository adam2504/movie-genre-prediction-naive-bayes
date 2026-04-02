import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Movie Genre Classifier", page_icon="🎬", layout="centered")

st.title("🎬 Movie Genre Classifier")
st.markdown(
    "Prédit le genre d'un film parmi **Animation / Horror / Drama** "
    "à partir de ses métadonnées numériques — Gaussian Naive Bayes."
)

# ── TODO: remplacer par le modèle entraîné depuis main.ipynb ──────────────────
# Le pipeline sera importé ici une fois exporté depuis main.ipynb (joblib/pickle)
# import joblib
# pipeline = joblib.load("model/pipeline.pkl")
st.warning("⚠️ Modèle non encore chargé — connecter à main.ipynb (voir TODO dans app.py)")

# ── Sidebar : saisie des features ─────────────────────────────────────────────
st.sidebar.header("Caractéristiques du film")

rating        = st.sidebar.slider("Note combinée (rating)", 0.0, 10.0, 6.5, step=0.1)
total_votes   = st.sidebar.number_input("Nombre total de votes", min_value=0, value=1000, step=100)
popularity    = st.sidebar.number_input("Popularité (TMDB)", min_value=0.0, value=10.0, step=1.0)
runtime       = st.sidebar.number_input("Durée (minutes)", min_value=0, value=100, step=5)
is_english    = st.sidebar.selectbox("Langue originale", ["Anglais", "Autre"]) == "Anglais"
cast_count    = st.sidebar.number_input("Nombre d'acteurs au casting", min_value=0, value=10, step=1)
release_month = st.sidebar.slider("Mois de sortie", 1, 12, 6)
release_year  = st.sidebar.slider("Année de sortie", 1900, 2025, 2010)
num_languages = st.sidebar.number_input("Nombre de langues parlées", min_value=0, value=1, step=1)
num_countries = st.sidebar.number_input("Nombre de pays de production", min_value=0, value=1, step=1)

# ── Prédiction ─────────────────────────────────────────────────────────────────
input_data = pd.DataFrame([{
    "rating":        rating,
    "total_votes":   total_votes,
    "popularity":    popularity,
    "runtime":       runtime,
    "is_english":    int(is_english),
    "cast_count":    cast_count,
    "release_month": release_month,
    "release_year":  release_year,
    "num_languages": num_languages,
    "num_countries": num_countries,
}])

st.subheader("Données saisies")
st.dataframe(input_data)

if st.button("Prédire le genre"):
    st.info("Modèle non encore connecté — implémentation à venir.")
    # Une fois le pipeline chargé :
    # genre = pipeline.predict(input_data)[0]
    # proba = pipeline.predict_proba(input_data)[0]
    # st.success(f"Genre prédit : **{genre}**")
    # for g, p in zip(pipeline.classes_, proba):
    #     st.progress(float(p), text=f"{g} : {p:.1%}")
