# Projet — Classification de Genre de Films (Naive Bayes)
**Cours : Apprentissage et Estimation Bayésienne — ING4 S8, ECE Paris**

---

## Objectif

Classifier le genre d'un film à partir de ses **métadonnées numériques uniquement** (pas de features textuelles dans un premier temps), en utilisant un modèle **Gaussian Naive Bayes**.

---

## Dataset

**Source :** [TMDB-IMDB Movies Dataset](https://huggingface.co/datasets/HenryWaltson/TMDB-IMDB-Movies-Dataset) — HuggingFace

Dataset combinant des données TMDB et IMDB, donnant accès à plus de **400 000 films** avec 29 colonnes (notes, popularité, budget, genres, casting, langues, etc.).

L'intérêt de combiner les deux sources est d'avoir à la fois un grand volume de données et une double évaluation des films (deux systèmes de notation distincts).

---

## Évolution du projet

### Étape 1 — Exploration et nettoyage

- Exploration classique du dataset (`.info()`, `.head()`, distributions)
- Suppression des **doublons**
- Suppression de colonnes inutiles pour l'instant : `backdrop_path`, `keywords`, `homepage`, `tconst`, `overview`, `poster_path`, `tagline` (features textuelles mises de côté)
- Conservation uniquement des films avec une **date de sortie renseignée** (colonne `release_date`), car cette information sera utilisée comme feature

> **Filtre budget/revenue (retiré) :** À une étape intermédiaire, un filtre avait été appliqué pour ne garder que les films avec des données de budget et revenue non nulles. Ce filtre a été **supprimé** car il éliminait ~380 000 entrées, réduisant le dataset à ~10 000 films — trop peu pour obtenir un bon modèle. Les colonnes `budget` et `revenue` ne sont donc pas utilisées comme features.

---

### Étape 2 — Feature Engineering

#### Fusion des notes TMDB et IMDB
Le dataset contient **4 colonnes de notation** distinctes :
- `vote_average` + `vote_count` (TMDB)
- `averageRating` + `numVotes` (IMDB)

Choix : calcul d'une **moyenne pondérée** par le nombre de votes pour obtenir une note unique plus robuste :

```
rating = (vote_average × vote_count + averageRating × numVotes) / (vote_count + numVotes)
total_votes = vote_count + numVotes
```

Les 4 colonnes originales sont ensuite supprimées.

#### Features retenues

| Feature | Description | Remarque |
|---|---|---|
| `rating` | Note combinée pondérée TMDB+IMDB | — |
| `total_votes` | Nombre total de votes combinés | — |
| `popularity` | Score de popularité TMDB | — |
| `runtime` | Durée du film (minutes) | — |
| `is_english` | Film en anglais ? (0/1) | Feature binaire |
| `cast_count` | Nombre d'acteurs au casting | À valider — dépend de la complétude des données |
| `release_month` | Mois de sortie | Saisonnalité potentielle |
| `release_year` | Année de sortie | — |
| `num_languages` | Nombre de langues parlées | À valider |
| `num_countries` | Nombre de pays de production | À valider |

> **Note :** Certaines de ces features sont à valider plus rigoureusement (tests statistiques, corrélations par genre). `cast_count`, `num_languages` et `num_countries` notamment dépendent de la complétude du dataset et pourraient être biaisiés.

---

### Étape 3 — Définition du label : le genre

#### Constat : multi-label
Le dataset contient **19 genres distincts**, mais la plupart des films en ont **plusieurs** (ex : *Interstellar* → `Adventure, Drama, Science Fiction`). Seulement ~185 000 films ont un genre unique.

#### Décision : prendre le premier genre
Pour simplifier en classification mono-label, le choix a été de prendre **uniquement le premier genre listé** pour chaque film.

> ⚠️ **Point à investiguer :** Cette décision est très dépendante de l'ordre établi par le dataset. On ne sait pas si cet ordre est arbitraire ou s'il reflète le genre "principal". Une alternative est en cours d'exploration dans `main copy.ipynb`.

---

### Étape 4 — Sélection des genres et undersampling

#### Distribution des genres (top)
| Genre | Nombre de films |
|---|---|
| Drama | 95 151 |
| Comedy | 62 676 |
| Documentary | 54 188 |
| Animation | 20 382 |
| Action | 20 179 |
| Horror | 19 401 |
| ... | ... |

#### Évolution des tentatives

| Étape | Genres | Accuracy | Macro F1 | Notes |
|---|---|---|---|---|
| Baseline | 5 genres (Drama, Comedy, Action, Horror, Adventure) | ~42% | ~0.28 | Fort déséquilibre des classes |
| + Undersampling | 5 genres | ~31% | ~0.26 | Biais corrigé, features insuffisantes |
| Simplification à 3 | Drama, Comedy, Action | ~44% | ~0.40 | Amélioration mais Comedy≈Drama numériquement |
| Retrait filtre budget/revenue | Drama, Comedy, Documentary | ~47% | ~0.39 | +200k films disponibles |
| Nouvelles features + ColumnTransformer | Drama, Comedy, Documentary | ~50% | ~0.45 | Objectif numérique atteint |
| **Meilleur combo trouvé** | **Animation, Horror, Drama** | **66%** | **0.66** | Genres numériquement distincts |

#### Pourquoi Animation / Horror / Drama ?
Ces trois genres ont des **profils numériques naturellement distincts**, ce qui les rend séparables par un modèle Naive Bayes :

- **Animation** : `cast_count` spécifique, `runtime` particulier, `release_year` plus ancien en moyenne, `rating` plus familial
- **Horror** : `popularity` variable, `runtime` court, faible `total_votes`, `release_year` récent
- **Drama** : `total_votes` élevé, `runtime` long, distribution de `rating` équilibrée

#### Pourquoi Comedy / Drama échouaient ?
Comedy et Drama partagent des distributions quasi identiques sur toutes les features numériques — leur distinction est **sémantique** (contenu narratif), pas quantitative. Un modèle sans features textuelles ne peut pas les différencier.

#### Undersampling
Pour éviter le biais de classe, on plafonne chaque genre au nombre de films du genre le moins représenté :
- ~19 401 films par genre → **58 203 films au total**
- Split 80/20 stratifié → 46 562 train / 11 641 test

---

### Étape 5 — Résultats du modèle final

**Modèle :** `GaussianNB` via `sklearn`, avec `RobustScaler` sur les features continues

```
              precision    recall  f1-score   support

   Animation       0.69      0.76      0.72      3880
       Drama       0.62      0.66      0.64      3880
      Horror       0.69      0.57      0.62      3881

    accuracy                           0.66     11641
   macro avg       0.66      0.66      0.66     11641
```

- **Animation** est le mieux prédit (F1 = 0.72) — profil le plus distinctif
- **Horror** a le plus faible recall (0.57) — confusion partielle avec Drama
- Priors équilibrés à **33.3%** chacun grâce à l'undersampling

---

## Structure du projet

```
Projet/
├── main.ipynb                          # Notebook actif — développement en cours
├── app.py                              # App Streamlit (squelette — à connecter au pipeline)
├── README.md
├── enonce.md
├── requirements.txt
└── notebooks/
    ├── baseline.ipynb                  # Pipeline propre de référence (ne pas modifier)
    ├── archive/
    │   └── brouillon.ipynb             # Ancien notebook brouillon (historique)
    ├── exploration/
    │   └── genre_label_strategy.ipynb  # Stratégies alternatives pour le label genre
    └── experiments/
        └── (copies de baseline pour tests isolés)
```

### Principe de travail
- **`main.ipynb`** : notebook actif, basé sur `baseline.ipynb`, où le développement continue
- **`baseline.ipynb`** : point de départ figé et propre — ne pas modifier
- **`experiments/`** : pour chaque test (ex: retirer une feature, changer les genres), copier `baseline.ipynb` ici avec un nom explicite et n'y faire **qu'un seul changement**
- **`app.py`** : app Streamlit qui sera connectée au pipeline exporté depuis `main.ipynb` (via `joblib`)

---

## Prochaines étapes

- [ ] Valider la décision "premier genre" vs autres stratégies (`notebooks/exploration/genre_label_strategy.ipynb`)
- [ ] Valider et justifier chaque feature numériquement (corrélations, tests) — notamment `cast_count`, `num_languages`, `num_countries`
- [ ] Tester l'ajout de **features textuelles (TF-IDF)** sur `overview`/`keywords` pour permettre la distinction de genres sémantiquement proches
- [ ] Envisager un retour à 4-5 genres avec les features textuelles
- [ ] Connecter `app.py` au pipeline exporté depuis `main.ipynb` (export `joblib`, ajout `streamlit` dans `requirements.txt`)
