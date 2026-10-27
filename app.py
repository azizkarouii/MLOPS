
# Importer Literal pour limiter les valeurs autorisées
from typing import Literal

# Importer MLflow pour charger le modèle enregistré
import mlflow
import mlflow.sklearn

# Importer Pandas pour préparer les données du client
import pandas as pd

# Importer FastAPI et Pydantic
from fastapi import FastAPI
from pydantic import BaseModel


# Créer l'application FastAPI
# Ce titre sera visible dans la documentation Swagger
app = FastAPI(title="API Churn")


# Charger le modèle champion une seule fois au démarrage
# L'alias permet de changer de version sans modifier cette ligne
mlflow.set_tracking_uri("sqlite:///mlflow.db")

model = mlflow.sklearn.load_model(
    "models:/churn-model@champion"
)


# Décrire les champs et les types des données reçues
# Pydantic validera automatiquement les requêtes
class Client(BaseModel):
    age: int
    salaire: float
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float
    consommation: float


# Route GET /health : vérifier que l'API fonctionne
@app.get("/health")
def health():
    return {"status": "ok"}


# Route POST /predict : recevoir un client et prédire son churn
@app.post("/predict")
def predict(client: Client):

    # Convertir les données validées en DataFrame d'une ligne
    ligne = pd.DataFrame([client.model_dump()])

    # Convertir la variable catégorielle abonnement en colonnes binaires
    ligne = pd.get_dummies(ligne)

    # Aligner les colonnes sur celles utilisées lors de l'entraînement
    # Les colonnes absentes sont créées avec la valeur 0
    ligne = ligne.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Prédire : 1 = le client part, 0 = il reste
    resultat = int(model.predict(ligne)[0])

    # Retourner la prédiction au format JSON
    return {
        "churn": resultat,
        "message": (
            "Le client va partir"
            if resultat == 1
            else "Le client reste"
        )
    }
