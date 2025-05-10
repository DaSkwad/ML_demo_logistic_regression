# Régression Logistique sur le dataset Pima Indians Diabetes

# Cellule Markdown: Introduction
"""
## Introduction

Ce notebook pédagogique a pour objectif de vous initier à l'utilisation de la **régression logistique** dans un contexte réel de classification binaire, à partir du célèbre dataset des **Pima Indians Diabetes**. 

Nous chercherons à prédire si une patiente présente un risque de diabète en fonction de caractéristiques médicales telles que son âge, son taux de glucose, sa pression artérielle, etc.

Nous allons suivre les étapes classiques d’un projet de Machine Learning : compréhension des données, traitement des valeurs manquantes, entraînement d’un modèle, et enfin évaluation de ses performances.
"""

# Cellule Markdown: Avant-propos
"""
## Avant-propos

Le dataset utilisé provient d’une étude menée sur des femmes d’origine amérindienne Pima âgées de plus de 21 ans. Il est souvent utilisé à des fins pédagogiques pour les problèmes de classification.

L’algorithme de régression logistique, bien qu’étant l’un des plus simples en apprentissage supervisé, est particulièrement efficace pour ce type de tâche binaire. Il est aussi facilement interprétable, ce qui en fait un excellent point de départ pour comprendre les modèles prédictifs.
"""

# Cellule 1: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Cellule Markdown: Chargement des données
"""
### 1. Chargement du dataset

Commençons par charger notre dataset et en afficher les premières lignes pour en comprendre la structure.
"""

# Cellule 2: Chargement des données
df = pd.read_csv("diabetes.csv")
df.head()

# Cellule Markdown: Analyse des valeurs manquantes
"""
### 2.1 Analyse des valeurs manquantes

Certaines colonnes peuvent contenir des valeurs aberrantes (par exemple des zéros là où ce n’est pas possible médicalement). Nous allons les identifier.
"""

# Cellule 3: Analyse des valeurs nulles ou invalides
print(df.isnull().sum())
print("\nValeurs aberrantes (0) dans certaines colonnes :")
print((df == 0).sum())

# Cellule Markdown: Remplacement des valeurs invalides
"""
### 2.2 Traitement des valeurs manquantes

Nous remplaçons les valeurs aberrantes (zéros) par des `NaN` dans certaines colonnes médicales, puis nous remplissons ces valeurs par la médiane de la colonne.
"""

# Cellule 4: Remplacement des 0 par NaN pour certaines colonnes
cols_with_zero_invalid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_zero_invalid] = df[cols_with_zero_invalid].replace(0, np.nan)

# Cellule 5: Traitement des valeurs manquantes
for col in cols_with_zero_invalid:
    df[col].fillna(df[col].median(), inplace=True)

# Cellule Markdown: Analyse exploratoire
"""
### 2.3 Analyse exploratoire

Quelques visualisations de base nous aideront à mieux comprendre la répartition des classes et la corrélation entre les variables.
"""

# Cellule 6: Analyse exploratoire rapide
sns.countplot(x='Outcome', data=df)
plt.title("Répartition des classes")
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Matrice de corrélation")
plt.show()

# Cellule Markdown: Séparation des variables
"""
### 3.1 Séparation des variables explicatives et de la cible

Nous séparons la variable cible (`Outcome`) des autres colonnes explicatives.
"""

# Cellule 7: Séparation des variables
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Cellule Markdown: Standardisation des données
"""
### 3.2 Standardisation des données

Nous utilisons un `StandardScaler` pour mettre à l'échelle nos données numériques. C'est une bonne pratique pour les modèles linéaires comme la régression logistique.
"""

# Cellule 8: Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cellule Markdown: Création des jeux de données
"""
### 3.3 Création des jeux d'entraînement et de test

Nous divisons nos données en un jeu d'entraînement (80%) et un jeu de test (20%).
"""

# Cellule 9: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Cellule Markdown: Entraînement du modèle
"""
### 3.4 Entraînement du modèle de régression logistique

Nous entraînons notre modèle `LogisticRegression` sur les données d'entraînement.
"""

# Cellule 10: Entraînement du modèle
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Cellule Markdown: Évaluation du modèle
"""
### 3.5 Évaluation du modèle

Nous évaluons les performances du modèle à l’aide de plusieurs métriques : `accuracy`, `classification report` et `matrice de confusion`.
"""

# Cellule 11: Évaluation du modèle
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Cellule 12: Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Prédiction")
plt.ylabel("Réalité")
plt.title("Matrice de confusion")
plt.show()

# Cellule Markdown: Interprétation des coefficients
"""
### 4.1 Interprétation des coefficients

La régression logistique permet d’interpréter l’importance des variables à travers les coefficients associés à chaque feature.
"""

# Cellule 13: Interprétation des coefficients
coeff_df = pd.DataFrame({
    "Feature": df.columns[:-1],
    "Coefficient": model.coef_[0]
})
coeff_df.sort_values(by="Coefficient", ascending=False, inplace=True)
print(coeff_df)

# Cellule Markdown: Sauvegarde du modèle
"""
### 5.1 Sauvegarde du modèle

Nous sauvegardons le modèle et le standardiseur pour une réutilisation ultérieure (inférence, API, etc).
"""

# Cellule 14: Sauvegarde du modèle et du scaler
joblib.dump(model, "logistic_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Cellule Markdown: Utilisation du modèle sauvegardé
"""
### 5.2 Utilisation du modèle sauvegardé

Voici comment recharger le modèle et faire une prédiction sur un nouvel exemple (donné arbitrairement).
"""

# Cellule 15: Rechargement et prédiction
demo_input = np.array([[5, 166, 72, 19, 175, 25.8, 0.587, 51]])  # exemple
scaler_loaded = joblib.load("scaler.pkl")
model_loaded = joblib.load("logistic_model.pkl")

scaled_input = scaler_loaded.transform(demo_input)
predicted_class = model_loaded.predict(scaled_input)
print("Prédiction (0 = pas de diabète, 1 = diabète):", predicted_class[0])
