import pandas as pd

from src.data.make_dataset import load_processed
from src.features.build_features import normlise_text, level
from src.models.cluster import calcul_silhouette, cluster_courses_bisecting_kmeans, determine_optimal_clusters_bisecting
from src.models.predict_model import cluster_courses,recommend_similar_courses,determine_optimal_clusters
from src.models.recommendation_model import recommend_similar_courses_test
from src.visualization.visualize import visualize_course_levels, visualize_clusters, visualize_kcluster, \
    word_cloud_text, visualise_clusters
from pyspark.sql.functions import col



def main():
    data = load_processed()
    data=data.filter(data["description"].isNotNull())
    print(data.printSchema())
    print(data.head())
    data=level(data)
    level_counts = data.groupBy("platform", "level").count()
    level_counts_pandas = level_counts.toPandas()
    #visualize_course_levels(level_counts_pandas)


    skills_df = normlise_text(data)
    word_cloud_text(skills_df)
    costs =determine_optimal_clusters_bisecting(skills_df)
    visualize_kcluster(costs)
    model, clustered_df_kmeans = cluster_courses(skills_df)
    silhouette_kmeans =calcul_silhouette(clustered_df_kmeans)
    print("silhouette of cluster courses kmeans :")
    print(silhouette_kmeans)
    visualize_clusters(clustered_df_kmeans)
    model, clustered_df = cluster_courses_bisecting_kmeans(skills_df)
    visualize_clusters(clustered_df)
    silhouette =calcul_silhouette(clustered_df)
    print("silhouette of cluster :")
    # Create a DataFrame with the silhouette scores
    silhouette_scores = pd.DataFrame({
      'Algorithm': ['KMeans', 'Bisecting KMeans'],
      'Silhouette Score': [silhouette_kmeans, silhouette]
     })
    print(silhouette_scores)
    course_title = input("Entrez le titre d'un cours pour obtenir des recommandations : ")

    recommended_courses = recommend_similar_courses_test(course_title, clustered_df_kmeans, 5)

    print("Cours recommandés :")
    for title in recommended_courses:
          print(title)

if __name__ == "__main__":
    main()
