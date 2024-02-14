from pyspark.sql.functions import col
def aggregate_levels(df):
    return df.groupBy("platform", "level").count()
