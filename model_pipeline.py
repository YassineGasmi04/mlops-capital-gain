"""Pipeline ML modulaire : prédire si une personne a des gains en capital.

Projet Adult Census (UCI) - Membre 6.
Utilisateur visé : un organisme d'éducation financière.
Cible : has_capital_gain = 1 si capital-gain > 0, sinon 0 (classification binaire).

Chaque étape du notebook est encapsulée dans une fonction réutilisable :
prepare_data, train_model, evaluate_model, save_model, load_model.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLONNES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]
CIBLE = "has_capital_gain"
VARIABLES_CATEGORIELLES = ["education_groupe", "tranche_age", "occupation", "workclass"]
VARIABLES_NUMERIQUES = ["est_marie", "hours-per-week"]
COLONNES_MANQUANTES = ["workclass", "occupation", "native-country"]
RANDOM_STATE = 42


def regrouper_education(niveau):
    """Regroupe les 16 niveaux d'études (education-num) en 5 groupes."""
    if niveau <= 8:
        return "1-Sans diplome secondaire"
    if niveau == 9:
        return "2-HS-grad"
    if niveau <= 12:
        return "3-Etudes sup. courtes"
    if niveau == 13:
        return "4-Bachelors"
    return "5-Master et plus"


def load_data(chemin):
    """Charge le fichier adult.data (sans en-tête) dans un DataFrame."""
    return pd.read_csv(chemin, header=None, names=COLONNES, skipinitialspace=True, na_values="?")


def validate_data(df):
    """Vérifie que les données sont exploitables avant l'entraînement.

    Lève une ValueError si une colonne attendue manque ou si la table est vide.
    """
    manquantes = [c for c in COLONNES if c not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes manquantes : {manquantes}")
    if df.empty:
        raise ValueError("Le jeu de données est vide.")
    if (df["capital-gain"] < 0).any():
        raise ValueError("capital-gain contient des valeurs négatives.")
    return True


def clean_data(df):
    """Supprime les doublons et remplace les valeurs manquantes par 'Unknown'."""
    df = df.drop_duplicates().copy()
    for col in COLONNES_MANQUANTES:
        df[col] = df[col].fillna("Unknown")
    return df


def add_features(df):
    """Crée la cible et les trois variables de l'objectif Data Science."""
    df = df.copy()
    df[CIBLE] = (df["capital-gain"] > 0).astype(int)
    df["tranche_age"] = pd.cut(
        df["age"],
        bins=[16, 25, 35, 45, 55, 65, 100],
        labels=["17-25", "26-35", "36-45", "46-55", "56-65", "66+"],
    ).astype(str)
    df["est_marie"] = (
        df["marital-status"].isin(["Married-civ-spouse", "Married-AF-spouse"]).astype(int)
    )
    df["education_groupe"] = df["education-num"].apply(regrouper_education)
    return df


def balance_data(df):
    """Équilibre les classes par sous-échantillonnage (50 % / 50 %).

    On garde toutes les personnes avec gains et on tire au hasard
    le même nombre de personnes sans gains.
    """
    avec_gains = df[df[CIBLE] == 1]
    sans_gains = df[df[CIBLE] == 0].sample(n=len(avec_gains), random_state=RANDOM_STATE)
    return pd.concat([avec_gains, sans_gains]).sample(frac=1, random_state=RANDOM_STATE)


def prepare_data(chemin, test_size=0.2):
    """Charge et prétraite les données, puis les sépare en entraînement et test.

    Étapes : chargement, validation, nettoyage, création des variables,
    équilibrage, sélection des caractéristiques et séparation 80 / 20.

    Retourne X_train, X_test, y_train, y_test.
    """
    df = load_data(chemin)
    validate_data(df)
    df = clean_data(df)
    df = add_features(df)
    df = balance_data(df)

    X = df[VARIABLES_CATEGORIELLES + VARIABLES_NUMERIQUES]
    y = df[CIBLE]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)


def build_model():
    """Construit le pipeline : One-Hot + normalisation + Random Forest."""
    preparation = ColumnTransformer(
        [
            ("texte", OneHotEncoder(handle_unknown="ignore"), VARIABLES_CATEGORIELLES),
            ("nombres", StandardScaler(), VARIABLES_NUMERIQUES),
        ]
    )
    modele = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preparation", preparation), ("modele", modele)])


def train_model(X_train, y_train):
    """Entraîne le pipeline Random Forest et le retourne."""
    modele = build_model()
    modele.fit(X_train, y_train)
    return modele


def evaluate_model(modele, X_test, y_test):
    """Évalue le modèle sur le jeu de test.

    Retourne un dictionnaire : accuracy (mesure principale), precision,
    recall, f1, roc_auc et la matrice de confusion.
    """
    predictions = modele.predict(X_test)
    probabilites = modele.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilites),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def save_model(modele, chemin="model.joblib"):
    """Sauvegarde le modèle entraîné avec joblib."""
    joblib.dump(modele, chemin)
    return chemin


def load_model(chemin="model.joblib"):
    """Charge un modèle sauvegardé avec joblib."""
    return joblib.load(chemin)


def predict(modele, profils):
    """Prédit 0 ou 1 pour une liste de profils (dictionnaires).

    Chaque profil contient : education_groupe, tranche_age, occupation,
    workclass, est_marie, hours-per-week.
    """
    X = pd.DataFrame(profils)[VARIABLES_CATEGORIELLES + VARIABLES_NUMERIQUES]
    return np.asarray(modele.predict(X))
