"""Orchestration du pipeline ML avec Prefect.

Tasks : installation des dépendances, formatage, qualité, sécurité, tests unitaires,
préparation des données, entraînement, sauvegarde, chargement, évaluation, prédiction.
Les tasks ML font appel aux fonctions du fichier main.py.

Flows :
    python pipeline_prefect.py --flow all        # code + données + modèle + prédiction
    python pipeline_prefect.py --flow code       # formatage, qualité, sécurité, tests
    python pipeline_prefect.py --flow train      # préparation + entraînement (+ sauvegarde)
    python pipeline_prefect.py --flow evaluate   # chargement du modèle + évaluation
    python pipeline_prefect.py --flow predict    # chargement du modèle + prédiction
"""

import argparse
import os
import subprocess  # nosec B404 : commandes fixes, sans entrée utilisateur
import sys

from prefect import flow, get_run_logger, task

import main

PYTHON = sys.executable
FICHIERS_PROJET = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]
DOSSIER_TESTS = "tests"


def _run(commande):
    """Exécute une commande, affiche sa sortie et lève une erreur si elle échoue."""
    logger = get_run_logger()
    logger.info("Commande : %s", " ".join(commande))
    resultat = subprocess.run(  # nosec B603 : liste d'arguments fixe, pas de shell
        commande, capture_output=True, text=True, check=False
    )
    if resultat.stdout.strip():
        logger.info(resultat.stdout.strip())
    if resultat.returncode != 0:
        logger.error(resultat.stderr.strip())
        raise RuntimeError(f"Échec de la commande : {' '.join(commande)}")
    return resultat.stdout


# ---------------------------------------------------------------- tasks : code
@task(name="fetch-project")
def fetch_project():
    """Récupère la dernière version du projet depuis le dépôt GitHub (git pull)."""
    if not os.path.isdir(".git"):
        get_run_logger().warning("Pas de dépôt git ici : récupération ignorée.")
        return
    _run(["git", "pull", "--ff-only"])
    get_run_logger().info("Projet à jour avec le dépôt distant.")


@task(name="install-dependencies")
def install_dependencies():
    """Installe les bibliothèques listées dans requirements.txt."""
    _run([PYTHON, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    get_run_logger().info("Dépendances installées.")


@task(name="format-code")
def format_code():
    """Formate le code avec black (uniquement les fichiers du projet)."""
    _run([PYTHON, "-m", "black", "--line-length", "100", *FICHIERS_PROJET, DOSSIER_TESTS])
    get_run_logger().info("Code formaté.")


@task(name="code-quality")
def check_quality():
    """Vérifie la qualité du code avec flake8 et pylint."""
    _run([PYTHON, "-m", "flake8", *FICHIERS_PROJET])
    _run([PYTHON, "-m", "pylint", *FICHIERS_PROJET])
    get_run_logger().info("Qualité du code vérifiée.")


@task(name="code-security")
def check_security():
    """Analyse la sécurité du code avec bandit (sévérité moyenne et haute)."""
    _run([PYTHON, "-m", "bandit", "-ll", "-q", *FICHIERS_PROJET])
    get_run_logger().info("Aucune faille de sécurité moyenne ou haute.")


@task(name="unit-tests")
def run_tests():
    """Exécute les tests unitaires avec pytest."""
    _run([PYTHON, "-m", "pytest", "-q", DOSSIER_TESTS])
    get_run_logger().info("Tests unitaires réussis.")


# ---------------------------------------------------------- tasks : données / modèle
@task(name="prepare-data")
def prepare_data():
    """Prépare les données (appel à main.etape_prepare)."""
    return main.etape_prepare(main.CHEMIN_DONNEES)


@task(name="train-model")
def train_model(X_train, y_train):
    """Entraîne le modèle (appel à main.etape_train)."""
    return main.etape_train(X_train, y_train)


@task(name="save-model")
def save_model(modele):
    """Sauvegarde le modèle (appel à main.etape_save)."""
    return main.etape_save(modele, main.CHEMIN_MODELE)


@task(name="load-model")
def load_model():
    """Charge le modèle sauvegardé (appel à main.etape_load)."""
    return main.etape_load(main.CHEMIN_MODELE)


@task(name="evaluate-model")
def evaluate_model(modele, X_test, y_test):
    """Évalue le modèle (appel à main.etape_evaluate)."""
    resultats = main.etape_evaluate(modele, X_test, y_test)
    get_run_logger().info("Accuracy sur le test : %.3f", resultats["accuracy"])
    return resultats


@task(name="predict")
def make_prediction(modele):
    """Prédit pour un profil exemple (appel à main.etape_predict)."""
    return main.etape_predict(modele)


# ---------------------------------------------------------------------- flows
@flow(name="code", log_prints=True)
def flow_code():
    """Dépendances, formatage, qualité, sécurité et tests unitaires."""
    fetch_project()
    install_dependencies()
    format_code()
    check_quality()
    check_security()
    run_tests()


@flow(name="train", log_prints=True)
def flow_train():
    """Préparation des données, entraînement et sauvegarde du modèle."""
    X_train, _, y_train, _ = prepare_data()
    modele = train_model(X_train, y_train)
    save_model(modele)


@flow(name="evaluate", log_prints=True)
def flow_evaluate():
    """Chargement du modèle sauvegardé et évaluation."""
    _, X_test, _, y_test = prepare_data()
    modele = load_model()
    return evaluate_model(modele, X_test, y_test)


@flow(name="predict", log_prints=True)
def flow_predict():
    """Chargement du modèle sauvegardé et prédiction."""
    return make_prediction(load_model())


@flow(name="all", log_prints=True)
def flow_all():
    """Pipeline complet : code, puis préparation, entraînement, sauvegarde,
    évaluation et prédiction."""
    flow_code()
    X_train, X_test, y_train, y_test = prepare_data()
    modele = train_model(X_train, y_train)
    save_model(modele)
    resultats = evaluate_model(modele, X_test, y_test)
    make_prediction(modele)
    return resultats


FLOWS = {
    "all": flow_all,
    "code": flow_code,
    "train": flow_train,
    "entrainement": flow_train,
    "evaluate": flow_evaluate,
    "predict": flow_predict,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline ML orchestré avec Prefect")
    parser.add_argument("--flow", choices=FLOWS.keys(), default="all", help="flow à exécuter")
    FLOWS[parser.parse_args().flow]()
