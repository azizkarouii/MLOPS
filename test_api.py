
# Importer requests pour envoyer des requêtes HTTP
import requests

# Définir l'adresse de notre API locale
URL = "http://localhost:8000"


# 1. Vérifier si l'API est en fonctionnement
response = requests.get(URL + "/health")

print("GET /health ->", response.json())


# 2. Tester un client avec une faible consommation
# et une faible ancienneté
client1 = {
    "age": 35,
    "salaire": 2500,
    "abonnement": "Standard",
    "anciennete": 1,
    "consommation": 10
}

# Envoyer les données au format JSON à /predict
response = requests.post(
    URL + "/predict",
    json=client1
)

print(
    "POST /predict (client 1) ->",
    response.status_code,
    response.json()
)


# 3. Tester un client fidèle
client2 = {
    "age": 50,
    "salaire": 4000,
    "abonnement": "Premium",
    "anciennete": 10,
    "consommation": 80
}

response = requests.post(
    URL + "/predict",
    json=client2
)

print(
    "POST /predict (client 2) ->",
    response.status_code,
    response.json()
)


# 4. Provoquer volontairement une erreur de validation
# Copier client1 en remplaçant l'âge par une chaîne invalide
mauvais = dict(client1, age="abc")

response = requests.post(
    URL + "/predict",
    json=mauvais
)

print(
    "POST /predict (age='abc') ->",
    response.status_code
)

# Afficher le message renvoyé par Pydantic
if response.status_code == 422:
    print("Message :", response.json()["detail"][0]["msg"])
