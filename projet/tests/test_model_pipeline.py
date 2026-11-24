"""Tests unitaires des fonctions de model_pipeline.py."""

import os

import pandas as pd
import pytest

from model_pipeline import (
    CIBLE,
    add_features,
    balance_data,
    prepare_data,
    regrouper_education,
    validate_data,
)

DONNEES = "adult.data"


def _exemple():
    """Petit DataFrame au format adult.data pour les tests."""
    return pd.DataFrame(
        {
            "age": [22, 40, 60, 35],
            "workclass": ["Private"] * 4,
            "fnlwgt": [1, 2, 3, 4],
            "education": ["HS-grad", "Bachelors", "Masters", "HS-grad"],
            "education-num": [9, 13, 14, 9],
            "marital-status": [
                "Never-married",
                "Married-civ-spouse",
                "Married-civ-spouse",
                "Divorced",
            ],
            "occupation": ["Sales"] * 4,
            "relationship": ["Own-child", "Husband", "Wife", "Unmarried"],
            "race": ["White"] * 4,
            "sex": ["Male", "Male", "Female", "Female"],
            "capital-gain": [0, 5000, 0, 1200],
            "capital-loss": [0, 0, 0, 0],
            "hours-per-week": [20, 45, 40, 38],
            "native-country": ["United-States"] * 4,
            "income": ["<=50K", ">50K", ">50K", "<=50K"],
        }
    )


def test_regrouper_education():
    """Les niveaux d'études sont bien regroupés en 5 groupes."""
    assert regrouper_education(5) == "1-Sans diplome secondaire"
    assert regrouper_education(9) == "2-HS-grad"
    assert regrouper_education(13) == "4-Bachelors"
    assert regrouper_education(16) == "5-Master et plus"


def test_validate_data_rejette_colonne_manquante():
    """validate_data lève une erreur si une colonne manque."""
    with pytest.raises(ValueError):
        validate_data(_exemple().drop(columns=["age"]))


def test_add_features_cree_la_cible():
    """La cible vaut 1 si capital-gain > 0, sinon 0."""
    df = add_features(_exemple())
    assert df[CIBLE].tolist() == [0, 1, 0, 1]
    assert df["est_marie"].tolist() == [0, 1, 1, 0]


def test_balance_data_equilibre_les_classes():
    """Après équilibrage, il y a autant de 0 que de 1."""
    df = balance_data(add_features(_exemple()))
    assert df[CIBLE].sum() == len(df) / 2


@pytest.mark.skipif(not os.path.exists(DONNEES), reason="adult.data absent")
def test_prepare_data_separe_80_20():
    """prepare_data renvoie un jeu de test de 20 % avec 6 variables."""
    X_train, X_test, _, _ = prepare_data(DONNEES)
    assert X_train.shape[1] == 6
    assert round(len(X_test) / (len(X_train) + len(X_test)), 1) == 0.2
