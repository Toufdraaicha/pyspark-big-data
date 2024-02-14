
from pyspark.mllib.linalg import DenseVector, SparseVector, Vectors
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.shell import spark
from pyspark.sql.connect.functions import monotonically_increasing_id
from pyspark.sql.types import DoubleType, FloatType
import numpy as np
from pyspark.sql.functions import col, udf, lit
from pyspark.ml.feature import Normalizer

# Enregistrez la UDF avec le type de retour correct et les types de vecteur

def recommend_similar_courses(course_title, courses_df, num_recommendations=5):
  # Filtrer les cours par le même cluster
    course_row = courses_df.filter(col("title") == course_title).select("features", "prediction").first()

    # Vérifier si le cours saisi a été trouvé
    if course_row is None:
        print(f"Aucun cours trouvé avec le titre : {course_title}")
        return []

    course_vec = course_row["features"]
    course_cluster = course_row["prediction"]
    cluster_courses = courses_df.filter(col("prediction") == course_cluster)
    # Convert the features column to an RDD of vectors
    vectors = cluster_courses.select("features").rdd.map(lambda row: Vectors.fromML(row.features))

# Create a RowMatrix from the vectors
    mat = RowMatrix(vectors)

# Compute cosine similarities
    similarity_matrix = mat.columnSimilarities()
    similarity_df = similarity_matrix.entries.toDF()

# Add a unique identifier to the original DataFrame
    rescaled_data = cluster_courses.withColumn("id", monotonically_increasing_id())

# Join the DataFrames on the identifier column
    joined_df = rescaled_data.join(similarity_df, rescaled_data.id == similarity_df.i)
    # Calculer la similarité avec les cours du même cluster


    top_courses_titles = joined_df.orderBy("similarity", ascending=False).select("title").rdd.flatMap(lambda r: r if r.title != course_title else []).collect()

    return top_courses_titles[:num_recommendations]
