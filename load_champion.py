
# Importer les bibliothèques nécessaires
import mlflow
import mlflow.sklearn
import pandas as pd

# Se connecter à la base MLflow du projet
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Charger le modèle grâce à son alias, sans connaître sa version
model = mlflow.sklearn.load_model(
    "models:/churn-model@champion"
)

print("Modele charge :", type(model).__name__)
print("Variables attendues :", list(model.feature_names_in_))

# Créer un exemple de client
# Les variables d'abonnement sont représentées en colonnes binaires
client = pd.DataFrame([{
    "age": 35,
    "salaire": 2500,
    "anciennete": 1,
    "consommation": 10,
    "abonnement_Premium": False,
    "abonnement_Standard": True
}])

# Réorganiser les colonnes dans l'ordre attendu par le modèle
# Cette opération sélectionne les variables réellement utilisées
client = client[list(model.feature_names_in_)]

# Effectuer la prédiction : 1 = départ, 0 = client qui reste
prediction = model.predict(client)[0]

print(
    "Le client va-t-il partir ? ->",
    "OUI" if prediction == 1 else "NON"
)
