import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Genre Classifier",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

RANDOM_STATE = 42
SELECTED_GENRES = ["Animation", "Horror", "Drama"]
continuous_features  = ["rating", "total_votes", "popularity"]
passthrough_features = ["is_english", "cast_count", "release_month", "release_year"]
features = continuous_features + passthrough_features

GENRE_COLORS  = {"Animation": "#4C72B0", "Horror": "#C44E52", "Drama": "#DD8452"}
GENRE_EMOJI   = {"Animation": "🎨", "Horror": "💀", "Drama": "🎭"}

FEATURE_LABELS = {
    "rating":        "Note combinée",
    "total_votes":   "Nombre de votes",
    "popularity":    "Popularité",
    "is_english":    "Film en anglais",
    "cast_count":    "Nombre d'acteurs",
    "release_month": "Mois de sortie",
    "release_year":  "Année de sortie",
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0e1117; }

    .hero {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        background: linear-gradient(90deg, #e0e7ff, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p { color: #94a3b8; font-size: 1rem; margin: 0; }

    .metric-card {
        background: #1e2433;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid rgba(255,255,255,0.06);
        text-align: center;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #a5b4fc;
    }
    .metric-card .label {
        font-size: 0.78rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }

    .result-card {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        border: 2px solid;
        margin-bottom: 1.5rem;
    }
    .result-genre {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .result-conf {
        font-size: 1rem;
        opacity: 0.75;
    }

    .section-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.5rem 0 0.75rem 0;
    }

    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
        cursor: pointer;
    }
    .stButton > button:hover { opacity: 0.9; }

    .sidebar-section {
        background: #1e2433;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.06);
    }

    [data-testid="stSidebar"] { background: #131720; }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #a5b4fc;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    div[data-testid="stSpinner"] { color: #a5b4fc; }
</style>
""", unsafe_allow_html=True)


# ── Données & modèle (cached) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement du dataset et entraînement du modèle…")
def load_and_train():
    df = pd.read_csv("hf://datasets/HenryWaltson/TMDB-IMDB-Movies-Dataset/TMDB  IMDB Movies Dataset.csv")
    df = df.drop_duplicates()
    df = df.drop(columns=["backdrop_path", "keywords", "homepage", "tconst",
                           "overview", "poster_path", "tagline"])
    df = df[df["release_date"].notna()]

    total_votes = df["vote_count"] + df["numVotes"]
    df["rating"]      = (df["vote_average"] * df["vote_count"] + df["averageRating"] * df["numVotes"]) / total_votes
    df["total_votes"] = total_votes
    df = df.drop(columns=["vote_count", "numVotes", "vote_average", "averageRating"])

    df["is_english"]    = (df["original_language"] == "en").astype(int)
    df["cast_count"]    = df["cast"].fillna("").apply(lambda x: len(x.split(",")) if x else 0)
    df["release_month"] = pd.to_datetime(df["release_date"]).dt.month
    df["release_year"]  = pd.to_datetime(df["release_date"]).dt.year

    df_clean = df[df["genres"].notna()].copy()
    df_clean["genre"] = df_clean["genres"].str.split(",").str[0].str.strip()
    df_sel = df_clean[df_clean["genre"].isin(SELECTED_GENRES)].copy()

    cap = df_sel["genre"].value_counts().min()
    df_balanced = df_sel.groupby("genre", group_keys=False).sample(n=cap, random_state=RANDOM_STATE)

    X_train, X_test, y_train, y_test = train_test_split(
        df_balanced[features], df_balanced["genre"],
        test_size=0.2, stratify=df_balanced["genre"], random_state=RANDOM_STATE,
    )

    preprocessor = ColumnTransformer([
        ("scale", RobustScaler(), continuous_features),
        ("pass",  "passthrough",  passthrough_features),
    ])
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", GaussianNB())])
    pipeline.fit(X_train, y_train)

    from sklearn.metrics import classification_report
    report = classification_report(y_test, pipeline.predict(X_test), output_dict=True)

    return pipeline, report, cap


def feature_contributions(pipeline, X_input):
    """
    Pour chaque feature, calcule sa contribution discriminante :
    log P(x_i | classe prédite) - moyenne des log P(x_i | classe) sur toutes les classes.
    Retourne un dict {feature: contribution} normalisé en % (contributions positives uniquement).
    """
    model = pipeline.named_steps["model"]
    X_transformed = pipeline.named_steps["preprocessor"].transform(X_input)

    classes = model.classes_
    pred_class_idx = np.argmax(model.predict_proba(X_transformed)[0])

    # Log-vraisemblances par feature et par classe : -0.5 * [(x - μ)² / σ² + log(2πσ²)]
    means = model.theta_          # (n_classes, n_features)
    variances = model.var_         # (n_classes, n_features)
    x = X_transformed[0]           # (n_features,)

    log_likelihoods = -0.5 * ((x - means) ** 2 / variances + np.log(2 * np.pi * variances))
    # log_likelihoods shape: (n_classes, n_features)

    # Contribution = log-vraisemblance pour la classe prédite - moyenne sur les classes
    contributions = log_likelihoods[pred_class_idx] - log_likelihoods.mean(axis=0)

    contrib_dict = {feat: float(c) for feat, c in zip(features, contributions)}
    return contrib_dict, classes[pred_class_idx]


# ── Chargement ────────────────────────────────────────────────────────────────
pipeline, report, cap = load_and_train()
macro_f1 = report["macro avg"]["f1-score"]
accuracy  = report["accuracy"]

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <h1>🎬 Movie Genre Classifier</h1>
    <p>Classifie un film parmi <strong>Animation · Horror · Drama</strong> à partir de ses métadonnées numériques — Gaussian Naive Bayes, entraîné sur {cap*3:,} films.</p>
</div>
""", unsafe_allow_html=True)

# ── Métriques modèle ──────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="value">{macro_f1:.1%}</div><div class="label">Macro F1</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="value">{accuracy:.1%}</div><div class="label">Accuracy</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="value">{cap*3:,}</div><div class="label">Films d\'entraînement</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="value">7</div><div class="label">Features</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Sidebar — inputs ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎥 Caractéristiques du film")

    st.markdown("**Notes & Popularité**")
    rating      = st.slider("Note combinée (0–10)", 0.0, 10.0, 6.5, step=0.1,
                             help="Moyenne pondérée TMDB + IMDB")
    total_votes = st.number_input("Nombre total de votes", min_value=0, value=1000, step=100)
    popularity  = st.number_input("Score de popularité TMDB", min_value=0.0, value=10.0, step=1.0)

    st.markdown("---")
    st.markdown("**Film**")
    is_english    = st.selectbox("Langue originale", ["Anglais", "Autre"]) == "Anglais"
    cast_count    = st.number_input("Nombre d'acteurs", min_value=0, value=6, step=1)

    st.markdown("---")
    st.markdown("**Sortie**")
    release_year  = st.slider("Année de sortie", 1920, 2025, 2005)
    release_month = st.slider("Mois de sortie", 1, 12, 6,
                               format="%d",
                               help="1 = Janvier, 12 = Décembre")

    st.markdown("---")
    predict_btn = st.button("🔮 Prédire le genre", use_container_width=True)

# ── Colonnes principales ──────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

input_df = pd.DataFrame([{
    "rating":        rating,
    "total_votes":   total_votes,
    "popularity":    popularity,
    "is_english":    int(is_english),
    "cast_count":    cast_count,
    "release_month": release_month,
    "release_year":  release_year,
}])

if predict_btn:
    input_data = pd.DataFrame([{
        "rating":        rating,
        "total_votes":   total_votes,
        "popularity":    popularity,
        "is_english":    int(is_english),
        "cast_count":    cast_count,
        "release_month": release_month,
        "release_year":  release_year,
    }])

    probas = pipeline.predict_proba(input_data)[0]
    classes = pipeline.classes_
    pred_genre = classes[np.argmax(probas)]
    pred_proba = np.max(probas)

    contrib_dict, _ = feature_contributions(pipeline, input_data)

    with col_left:
        # ── Résultat principal ────────────────────────────────────────────────
        color = GENRE_COLORS[pred_genre]
        emoji = GENRE_EMOJI[pred_genre]
        st.markdown(f"""
        <div class="result-card" style="background: {color}18; border-color: {color};">
            <div class="result-genre" style="color: {color};">{emoji} {pred_genre}</div>
            <div class="result-conf" style="color: {color};">Confiance : {pred_proba:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probabilités par genre ────────────────────────────────────────────
        st.markdown('<div class="section-title">Probabilités par genre</div>', unsafe_allow_html=True)

        fig_proba = go.Figure()
        sorted_idx = np.argsort(probas)[::-1]
        for i in sorted_idx:
            g = classes[i]
            fig_proba.add_trace(go.Bar(
                x=[probas[i]],
                y=[f"{GENRE_EMOJI[g]} {g}"],
                orientation="h",
                marker_color=GENRE_COLORS[g],
                marker_line_width=0,
                text=f"{probas[i]:.1%}",
                textposition="outside",
                showlegend=False,
            ))
        fig_proba.update_layout(
            xaxis=dict(range=[0, 1.15], showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0", size=13),
            margin=dict(l=0, r=40, t=10, b=10),
            height=160,
            bargap=0.3,
        )
        st.plotly_chart(fig_proba, use_container_width=True, config={"displayModeBar": False})

    with col_right:
        # ── Contribution des features ─────────────────────────────────────────
        st.markdown('<div class="section-title">Poids de chaque feature dans la prédiction</div>', unsafe_allow_html=True)
        st.caption(f"Contribution à la prédiction **{pred_genre}** — valeur positive = renforce la décision")

        # Normaliser pour afficher en %
        raw_values = np.array([contrib_dict[f] for f in features])
        total_pos = raw_values[raw_values > 0].sum()
        pct_values = np.where(raw_values > 0, raw_values / total_pos * 100, raw_values / total_pos * 100)

        feat_df = pd.DataFrame({
            "feature": [FEATURE_LABELS[f] for f in features],
            "contribution": raw_values,
            "pct": pct_values,
        }).sort_values("contribution", ascending=True)

        colors = [GENRE_COLORS[pred_genre] if v >= 0 else "#475569" for v in feat_df["contribution"]]

        fig_contrib = go.Figure()
        fig_contrib.add_trace(go.Bar(
            x=feat_df["contribution"],
            y=feat_df["feature"],
            orientation="h",
            marker_color=colors,
            marker_line_width=0,
            text=[f"{v:+.1f}" for v in feat_df["contribution"]],
            textposition="outside",
            showlegend=False,
        ))
        fig_contrib.add_vline(x=0, line_color="#475569", line_width=1)
        fig_contrib.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0", size=12),
            margin=dict(l=10, r=60, t=10, b=10),
            height=280,
            bargap=0.25,
        )
        st.plotly_chart(fig_contrib, use_container_width=True, config={"displayModeBar": False})

        # ── Table de contribution en % ────────────────────────────────────────
        total_abs = np.abs(raw_values).sum()
        pct_abs = np.abs(raw_values) / total_abs * 100
        feat_table = pd.DataFrame({
            "Feature": [FEATURE_LABELS[f] for f in features],
            "Valeur saisie": [
                f"{rating:.1f}", f"{total_votes:,}", f"{popularity:.1f}",
                "Oui" if is_english else "Non", str(cast_count),
                str(release_month), str(release_year),
            ],
            "Poids (%)": [f"{p:.1f}%" for p in pct_abs],
            "Direction": ["✅ Pour" if v >= 0 else "❌ Contre" for v in raw_values],
        }).sort_values("Poids (%)", ascending=False).reset_index(drop=True)

        st.dataframe(
            feat_table,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Poids (%)": st.column_config.ProgressColumn(
                    "Poids (%)", min_value=0, max_value=100, format="%.1f%%"
                )
            }
        )

else:
    # ── État initial ──────────────────────────────────────────────────────────
    with col_left:
        st.markdown('<div class="section-title">À propos du modèle</div>', unsafe_allow_html=True)

        model_info = {
            "Animation": report["Animation"],
            "Drama":     report["Drama"],
            "Horror":    report["Horror"],
        }
        rows = []
        for g, r in model_info.items():
            rows.append({
                "Genre": f"{GENRE_EMOJI[g]} {g}",
                "Précision": f"{r['precision']:.0%}",
                "Rappel":    f"{r['recall']:.0%}",
                "F1":        f"{r['f1-score']:.2f}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        st.markdown('<div class="section-title">Comment lire les contributions ?</div>', unsafe_allow_html=True)
        st.markdown("""
        Pour chaque prédiction, on calcule la **log-vraisemblance** de chaque feature sous GaussianNB :

        > **Contribution** = log P(x_i | genre prédit) − moyenne des log P(x_i | tous genres)

        - **Valeur positive** → cette feature "pousse" vers le genre prédit
        - **Valeur négative** → cette feature va à l'encontre de la prédiction
        - **Le poids (%)** représente l'importance relative de chaque feature dans la décision finale
        """)

    with col_right:
        st.markdown('<div class="section-title">Profils moyens par genre</div>', unsafe_allow_html=True)
        profiles = {
            "Feature": ["Note", "Votes", "Popularité", "% Anglais", "Acteurs", "Année"],
            "Animation": ["6.44", "2 694", "2.78", "63%", "2.6", "1985"],
            "Drama":     ["6.30", "2 933", "2.24", "37%", "7.6", "1994"],
            "Horror":    ["5.06", "4 175", "3.64", "73%", "6.8", "2006"],
        }
        st.dataframe(pd.DataFrame(profiles), hide_index=True, use_container_width=True)

        st.markdown('<div class="section-title">Remplissez les données dans la barre latérale</div>', unsafe_allow_html=True)
        st.info("Ajustez les caractéristiques du film dans le panneau gauche, puis cliquez sur **Prédire le genre**.")
