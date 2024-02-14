import seaborn as sns
from pyspark.sql.functions import explode, col, udf
import matplotlib
from pyspark.sql.types import StringType
from wordcloud import WordCloud
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
def visualize_course_levels(level_counts_pandas):
    """
    Visualise la répartition des niveaux de cours entre les différentes plateformes.

    :param level_counts_pandas: DataFrame Pandas contenant les données agrégées.
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(x="level", y="count", hue="platform", data=level_counts_pandas)
    plt.title("Comparaison du Nombre de Cours par Niveau entre les Plateformes")
    plt.xlabel("Niveau")
    plt.ylabel("Nombre de Cours")
    plt.show()
    plt.savefig('myplot.png')
def visualize_data(recommendations):
    """
    Visualise les recommandations de cours.

    :param recommendations: Dictionnaire de recommandations.
    """
    for course, similar_courses in recommendations.items():
        print(f"Recommandations pour le cours '{course}':")
        for similar_course in similar_courses:
            print(f" - {similar_course}")
        print("\n")

# Function to extract and count skills in a cluster

def visualize_kcluster(costs):
# Afficher le graphique du coude
   plt.plot(range(2, len(costs)+2), costs)
   plt.xlabel('Nombre de Clusters')
   plt.ylabel('Coût')
   plt.title('Méthode du Coude pour K-means')
   plt.show()
   plt.savefig('cluster.png')



def extract_skills_from_cluster(clustered_data,cluster_id):
    cluster_skills = clustered_data.filter(clustered_data.prediction == cluster_id)
    word_freq = cluster_skills.withColumn("word", explode(col("filtered_words"))).groupBy("word").count()
    return word_freq.orderBy(col("count").desc())


def visualize_clusters(clustered_data):
    for i in range(3):  # Assuming 5 clusters
        skills_df = extract_skills_from_cluster(clustered_data,i).toPandas().head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='count', y='word', data=skills_df)
        plt.title(f'Top 10 Skills in Cluster {i}')
        plt.xlabel('Frequency')
        plt.ylabel('Skills')
        plt.show()
        plt.savefig(f'skills{i}.png')

def word_cloud_text(courses):
    # Définir une UDF pour convertir une liste en chaîne de caractères
    list_to_string_udf = udf(lambda lst: ' '.join(lst), StringType())

# Appliquer l'UDF pour convertir les listes en chaînes
    courses = courses.withColumn("filtered_words_str", list_to_string_udf("filtered_words"))
    word_cloud_text = ' '.join(courses.select("filtered_words_str").rdd.flatMap(lambda x: x).collect())

    print(type(word_cloud_text))
    wordcloud = WordCloud(
       max_font_size=100,
       max_words=100,
       background_color="white",
       scale=10,
       width=800,
       height=800
     ).generate(word_cloud_text)

    plt.figure(figsize=(10,10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def visualise_clusters(local_data, k):
    plt.figure(figsize=(10, 6))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Ajoutez plus de couleurs si nécessaire
    local_data = local_data.toPandas()
    # Utiliser un indice comme axe Y pour la visualisation
    local_data['index'] = range(len(local_data))

    for i in range(k):
        cluster_data = local_data[local_data['prediction'] == i]
        plt.scatter(cluster_data['index'], cluster_data['title'], s=50, c=colors[i % len(colors)], label=f'Cluster {i + 1}')

    plt.title('Course Clusters')
    plt.xlabel('Index')
    plt.ylabel('Your Feature')
    plt.legend()
    plt.show()


