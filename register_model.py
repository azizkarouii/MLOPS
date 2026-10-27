
# Importer MLflow et son client pour gérer le Registry
import mlflow
from mlflow import MlflowClient

# Utiliser la même base de données que celle du TP3
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Récupérer les runs de l'expérience et les trier par F1 décroissant
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

# Sélectionner le meilleur run et récupérer son identifiant
best = runs.iloc[0]
run_id = best["run_id"]

print(
    "Meilleur run :",
    best["tags.mlflow.runName"],
    "| id :",
    run_id[:8],
    "..."
)

# Enregistrer le modèle du run sous le nom churn-model
# Une nouvelle version est créée à chaque enregistrement
result = mlflow.register_model(
    f"runs:/{run_id}/model",
    "churn-model"
)

print("Modele enregistre : churn-model | version", result.version)

# Créer l'alias champion et le faire pointer vers cette version
client = MlflowClient()
client.set_registered_model_alias(
    "churn-model",
    "champion",
    result.version
)

print("Alias 'champion' -> version", result.version)
