"""Point d'entrée : exécute les étapes du pipeline via des arguments CLI.

Chaque étape est aussi une fonction réutilisable, appelée par pipeline_prefect.py.

Exemples :
    python main.py --prepare
    python main.py --train
    python main.py --evaluate
    python main.py --predict
    python main.py --all
"""

import argparse

from model_pipeline import (
    evaluate_model,
    load_model,
    predict,
    prepare_data,
    save_model,
    train_model,
)

CHEMIN_DONNEES = "adult.data"
CHEMIN_MODELE = "model.joblib"
PROFIL_EXEMPLE = {
    "education_groupe": "5-Master et plus",
    "tranche_age": "46-55",
    "occupation": "Exec-managerial",
    "workclass": "Private",
    "est_marie": 1,
    "hours-per-week": 45,
}


def etape_prepare(donnees=CHEMIN_DONNEES):
    """Prépare les données, affiche leur taille et retourne les 4 jeux."""
    X_train, X_test, y_train, y_test = prepare_data(donnees)
    print(f"Entraînement : {X_train.shape} | Test : {X_test.shape}")
    print(f"Taux de positifs (train) : {y_train.mean():.2%}")
    return X_train, X_test, y_train, y_test


def etape_train(X_train, y_train):
    """Entraîne le modèle et le retourne."""
    modele = train_model(X_train, y_train)
    print("Modèle entraîné.")
    return modele


def etape_save(modele, modele_path=CHEMIN_MODELE):
    """Sauvegarde le modèle et retourne son chemin."""
    chemin = save_model(modele, modele_path)
    print(f"Modèle sauvegardé dans {chemin}")
    return chemin


def etape_load(modele_path=CHEMIN_MODELE):
    """Charge le modèle sauvegardé et le retourne."""
    modele = load_model(modele_path)
    print(f"Modèle chargé depuis {modele_path}")
    return modele


def etape_evaluate(modele, X_test, y_test):
    """Évalue le modèle, affiche les métriques et les retourne."""
    resultats = evaluate_model(modele, X_test, y_test)
    print("Résultats sur le jeu de test :")
    for nom in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"  {nom:<10}: {resultats[nom]:.3f}")
    print(f"  matrice de confusion : {resultats['confusion_matrix']}")
    return resultats


def etape_predict(modele, profil=None):
    """Prédit pour un profil (par défaut le profil exemple) et retourne 0 ou 1."""
    profil = profil or PROFIL_EXEMPLE
    resultat = int(predict(modele, [profil])[0])
    if resultat == 1:
        texte = "a probablement des gains en capital"
    else:
        texte = "n'a probablement pas de gains"
    print(f"Profil de test : {profil}")
    print(f"Prédiction : {resultat} -> la personne {texte}")
    return resultat


def main():
    """Lit les arguments de la ligne de commande et lance les étapes demandées."""
    parser = argparse.ArgumentParser(description="Pipeline ML - gains en capital (Adult)")
    parser.add_argument("--data", default=CHEMIN_DONNEES, help="chemin du fichier adult.data")
    parser.add_argument("--model", default=CHEMIN_MODELE, help="chemin du modèle sauvegardé")
    parser.add_argument("--prepare", action="store_true", help="préparer les données")
    parser.add_argument("--train", action="store_true", help="entraîner et sauvegarder le modèle")
    parser.add_argument("--evaluate", action="store_true", help="évaluer le modèle sauvegardé")
    parser.add_argument("--predict", action="store_true", help="prédire pour un profil exemple")
    parser.add_argument("--all", action="store_true", help="exécuter toutes les étapes")
    args = parser.parse_args()

    if not any([args.prepare, args.train, args.evaluate, args.predict, args.all]):
        parser.print_help()
        return

    X_train, X_test, y_train, y_test = etape_prepare(args.data)
    if args.train or args.all:
        etape_save(etape_train(X_train, y_train), args.model)
    if args.evaluate or args.all:
        etape_evaluate(etape_load(args.model), X_test, y_test)
    if args.predict or args.all:
        etape_predict(etape_load(args.model))


if __name__ == "__main__":
    main()
