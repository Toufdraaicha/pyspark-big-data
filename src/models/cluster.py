import numpy as np
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


def cluster_courses(df, features_col="features", k=3):
    # Définition du modèle K-means
    kmeans = KMeans().setK(k).setFeaturesCol(features_col)

    # Entraînement du modèle
    model = kmeans.fit(df)

    # Prédiction des clusters
    clustered_df = model.transform(df)
    return model, clustered_df


def determine_optimal_clusters(df, max_k=10):
    silhouette_scores = []

    for k in range(2, max_k+1):
        # Définition du modèle K-means
        kmeans = KMeans().setK(k).setFeaturesCol("features").setSeed(1)

        # Entraînement du modèle
        model = kmeans.fit(df)

        # Prédiction des clusters
        predictions = model.transform(df)

        # Évaluation avec le score de silhouette
        evaluator = ClusteringEvaluator()
        silhouette = evaluator.evaluate(predictions)
        silhouette_scores.append((k, silhouette))

    return silhouette_scores

from pyspark.ml.clustering import BisectingKMeans

def cluster_courses_bisecting_kmeans(df, features_col="features", k=3):
    # Définition du modèle Bisecting K-Means
    bkm = BisectingKMeans().setK(k).setFeaturesCol(features_col)

    # Entraînement du modèle
    model = bkm.fit(df)

    # Prédiction des clusters
    clustered_df = model.transform(df)
    return model, clustered_df


def determine_optimal_clusters_bisecting(df, max_k=10):
    silhouette_scores = []

    for k in range(2, max_k+1):
        # Définition du modèle K-means
         bkm = BisectingKMeans().setK(k).setFeaturesCol("features")

    # Entraînement du modèle
         model = bkm.fit(df)

        # Prédiction des clusters
         predictions = model.transform(df)

        # Évaluation avec le score de silhouette
         evaluator = ClusteringEvaluator()
         silhouette = evaluator.evaluate(predictions)
         silhouette_scores.append((k, silhouette))

    return silhouette_scores
def calcul_silhouette(predictions):
  evaluator = ClusteringEvaluator()
  silhouette = evaluator.evaluate(predictions)
  return silhouette
