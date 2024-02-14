from pyspark.sql.functions import col,concat,lit

def unify_dataframes(coursera_df, edx_df, udemy_df,common_columns):

    # Ajouter une colonne 'platform' à chaque DataFrame
    coursera_df = coursera_df.withColumnRenamed('course', 'title').withColumn('description', concat(col('skills'))).withColumn('platform', lit('cousera')).select(common_columns)
    edx_df = edx_df.withColumn('description',col('associatedskills')).withColumn('platform', lit('edX')).select(common_columns)
    udemy_df = udemy_df.withColumn('platform', lit('Udemy')).select(common_columns)

    # Fusionner tous les DataFrames en un seul
    unified_df = coursera_df.union(edx_df).union(udemy_df)

    return unified_df
