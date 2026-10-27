
# Importer MLflow et le client de gestion du Registry
import mlflow
from mlflow import MlflowClient

# Utiliser la base de données du projet
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# Retrouver le run essai-2, moins performant dans le TP3
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    filter_string="tags.mlflow.runName = 'essai-2'"
)

run_id = runs.iloc[0]["run_id"]

# Enregistrer ce modèle : une nouvelle version de churn-model est créée
version = mlflow.register_model(
    f"runs:/{run_id}/model",
    "churn-model"
)

print("Nouvelle version creee :", version.version)

# Sauvegarder la version actuellement désignée par champion
actuel = client.get_model_version_by_alias(
    "churn-model",
    "champion"
)

print("Champion actuel : version", actuel.version)

# Déplacer temporairement champion vers la nouvelle version
client.set_registered_model_alias(
    "churn-model",
    "champion",
    version.version
)

nouveau_champion = client.get_model_version_by_alias(
    "churn-model",
    "champion"
)

print(
    "Apres deplacement : champion -> version",
    nouveau_champion.version
)

# Restaurer l'alias sur la version initiale
client.set_registered_model_alias(
    "churn-model",
    "champion",
    actuel.version
)

champion_restaure = client.get_model_version_by_alias(
    "churn-model",
    "champion"
)

print(
    "Apres retour : champion -> version",
    champion_restaure.version
)
