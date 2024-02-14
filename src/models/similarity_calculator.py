from pyspark.ml.stat import Correlation

def calculate_similarity(df, features_col="features"):
    vector_col = "sfeatures"
    df = df.withColumn(vector_col, df[features_col].cast("vector"))
    corr = Correlation.corr(df, vector_col)
    return corr.collect()[0][0]
