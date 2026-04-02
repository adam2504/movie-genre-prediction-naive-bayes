# Projet — Classification de Genre de Films (Naive Bayes)
**Cours : Apprentissage et Estimation Bayésienne — ING4 S8, ECE Paris**

---

## Objectif

Classifier le genre d'un film à partir de ses **métadonnées numériques uniquement** (pas de features textuelles), en utilisant un modèle **Gaussian Naive Bayes**.

---

## Dataset

**Source :** [TMDB-IMDB Movies Dataset](https://huggingface.co/datasets/HenryWaltson/TMDB-IMDB-Movies-Dataset) — HuggingFace

Dataset combinant TMDB et IMDB : **400 000+ films**, 29 colonnes (notes, popularité, budget, genres, casting, langues, etc.).

---

## Évolution du projet

### Étape 1 — Exploration et nettoyage

- Exploration du dataset (`.info()`, `.head()`, distributions)
- Suppression des doublons et colonnes inutiles (`backdrop_path`, `keywords`, `homepage`, `tconst`, `overview`, `poster_path`, `tagline`)
- Conservation uniquement des films avec `release_date` renseignée

> **Filtre budget/revenue (retiré) :** Ce filtre éliminait ~380 000 entrées → dataset réduit à ~10 000 films, trop peu. Les colonnes `budget` et `revenue` ne sont pas utilisées.

---

### Étape 2 — Feature Engineering

#### Fusion des notes TMDB + IMDB

```
rating      = (vote_average × vote_count + averageRating × numVotes) / (vote_count + numVotes)
total_votes = vote_count + numVotes
```

#### Features retenues (7 features finales, validées par ablation)

| Feature | Type | Description |
|---|---|---|
| `rating` | Continue | Note combinée pondérée TMDB+IMDB |
| `total_votes` | Continue | Nombre total de votes combinés |
| `popularity` | Continue | Score de popularité TMDB |
| `is_english` | Binaire | Film en anglais ? (0/1) |
| `cast_count` | Discret | Nombre d'acteurs au casting |
| `release_month` | Discret | Mois de sortie |
| `release_year` | Discret | Année de sortie |

**Features retirées par ablation** (`notebooks/experiments/ablation_features.ipynb`) :
- `runtime` — suppression sans perte (Macro F1 stable)
- `num_languages` — idem
- `num_countries` — idem
- `budget` / `revenue` — catastrophique (NaN massifs → 33% accuracy)

---

### Étape 3 — Stratégie de label genre

Le dataset contient **19 genres**, mais la plupart des films en ont plusieurs. Pour un classifieur mono-label, 4 stratégies ont été testées (`notebooks/exploration/genre_label_strategy.ipynb`) :

| Stratégie | Macro F1 | N films |
|---|---|---|
| **S0 — Premier genre listé** ✓ | **0.67** | 58k |
| S2 — Priorité fixe (Animation > Horror > Drama) | 0.63 | 75k |
| S3 — Genre le plus rare | 0.63 | 75k |
| S4 — Multi-instance (un film → plusieurs lignes) | 0.63 | 75k |

**Décision : garder S0 (premier genre).** Aucune alternative ne bat la baseline. L'ordre TMDB est corrélé au genre principal du film.

Contexte complémentaire (`notebooks/exploration/single_genre_analysis.ipynb`) : le modèle performe mieux sur les films à genre unique (Macro F1 0.67, acc 75%) que sur les multi-genres (acc 68%), mais entraîner uniquement sur les films à genre unique ne change pas les performances globales — on garde donc tous les films.

---

### Étape 4 — Sélection des genres et undersampling

Toutes les combinaisons de 3 genres parmi les 19 disponibles ont été testées (`notebooks/experiments/genre_combination.ipynb`) — **969 combinaisons**.

**Résultat : Animation / Horror / Drama est dans le top 2% (rang 15/969)**, avec le **meilleur ratio performance / volume de données** :

| Combo | Macro F1 | Cap/genre |
|---|---|---|
| Animation, Thriller, Western | 0.762 | 4 166 |
| Animation, Horror, Western | 0.746 | 4 166 |
| ... | ... | ... |
| **Animation, Horror, Drama** ✓ | **0.674** | **19 401** |

Les combos "mieux classés" ont ~4 000 films/genre vs 19 401 pour notre choix — 5x moins de données, moins robustes.

**Undersampling :** on plafonne au genre le moins représenté :
- 19 401 films/genre → **58 203 films au total**
- Split 80/20 stratifié → 46 562 train / 11 641 test

#### Pourquoi ces 3 genres sont séparables

D'après les tests statistiques (`notebooks/experiments/feature_behavior.ipynb`) — Kruskal-Wallis, p ≈ 0 pour toutes les features :

| Feature | H-stat | Observation clé |
|---|---|---|
| `cast_count` | 17 199 | Animation ~2.6 acteurs vs Drama/Horror ~7 |
| `rating` | 11 198 | Horror note ~5.1 vs Animation/Drama ~6.4 |
| `is_english` | 5 645 | Drama peu anglophone (0.37) vs Horror (0.73) |
| `release_year` | 5 192 | Animation plus ancien (~1985) vs Horror récent (~2006) |
| `popularity` | 2 061 | Horror plus populaire (~3.6) vs Drama (~2.2) |
| `total_votes` | 1 976 | Horror plus voté (~4175) vs Animation (~2694) |
| `release_month` | 173 | Significatif mais moins discriminant |

---

### Étape 5 — Résultats du modèle final

**Modèle :** `GaussianNB` avec `RobustScaler` sur les 3 features continues, passthrough pour les 4 features discrètes.

**Cross-validation 5-fold stratifiée :**

```
Macro F1 : 0.673 ± 0.003
Accuracy : 0.674 ± 0.003
```

**Rapport de classification (test set) :**

```
              precision    recall  f1-score   support

   Animation       0.73      0.73      0.73      3880
       Drama       0.60      0.72      0.66      3880
      Horror       0.71      0.57      0.63      3881

    accuracy                           0.67     11641
   macro avg       0.68      0.67      0.67     11641
```

- **Animation** est le mieux prédit (F1 = 0.73) — profil le plus distinctif
- **Horror** a le plus faible recall (0.57) — confusion partielle avec Drama
- Priors équilibrés à **33.3%** chacun grâce à l'undersampling

---

## Structure du projet

```
Projet/
├── main.ipynb                                  # Notebook actif — pipeline final documenté
├── app.py                                      # App Streamlit (squelette — à connecter)
├── README.md
├── enonce.md
├── requirements.txt
└── notebooks/
    ├── baseline.ipynb                          # Pipeline propre de référence (ne pas modifier)
    ├── archive/
    │   └── brouillon.ipynb                     # Ancien notebook brouillon (historique)
    ├── exploration/
    │   ├── genre_label_strategy.ipynb          # Test des stratégies S0/S2/S3/S4 → S0 gagne
    │   └── single_genre_analysis.ipynb         # Films genre unique vs multi-genres
    └── experiments/
        ├── ablation_features.ipynb             # Sélection des 7 features finales
        ├── genre_combination.ipynb             # Test des 969 combos de 3 genres → top 2%
        └── feature_behavior.ipynb              # Distributions, séparabilité, gaussianité, corrélations
```

### Principe de travail
- **`main.ipynb`** : notebook actif avec toutes les décisions justifiées
- **`baseline.ipynb`** : point de départ figé — ne pas modifier
- **`experiments/`** : tests isolés, chacun un seul changement par rapport à baseline
- **`exploration/`** : analyse des données et choix méthodologiques
- **`app.py`** : app Streamlit à connecter au pipeline exporté via `joblib`

---

## Prochaines étapes

- [ ] Connecter `app.py` au pipeline exporté depuis `main.ipynb` (export `joblib`)
- [ ] PowerPoint 3 slides : contexte, problème, résultats
