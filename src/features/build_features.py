from pyspark.ml.feature import Tokenizer

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType


def extract_features(df, input_col="description", output_col="features"):
    tokenizer = Tokenizer(inputCol=input_col, outputCol="words")
    words_data = tokenizer.transform(df)

    hashing_tf = HashingTF(inputCol="words", outputCol="rawFeatures")
    featurized_data = hashing_tf.transform(words_data)

    idf = IDF(inputCol="rawFeatures", outputCol=output_col)
    idf_model = idf.fit(featurized_data)

    return idf_model.transform(featurized_data)


def normlise_text(df, input_column="description"):
   df = df.filter(df["description"].isNotNull())
   # Tokenization
   tokenizer = RegexTokenizer(inputCol="description", outputCol="words", pattern="\\W")
   tokenized_data = tokenizer.transform(df)

   remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
   filtered_data = remover.transform(tokenized_data)

   hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features")
   featurized_data = hashing_tf.transform(filtered_data)
   idf = IDF(inputCol="raw_features", outputCol="features")
   rescaled_data = idf.fit(featurized_data).transform(featurized_data)
   return rescaled_data

def normalize_level(level):
    mapping = {
        "Beginner": "Introductory",
        "Beginner Level": "Introductory",
        "Intermediate": "Intermediate",
        "Advanced": "Advanced",
        "Expert|":"Expert",
        "Specialization":"Expert",
        "Mixed":"All levels",
        "All Levels":"All levels"
        # Ajoutez ou modifiez les mappages en fonction des niveaux trouvés
    }
    return mapping.get(level, "Other")

normalize_level_udf = udf(normalize_level, StringType())

def level(data):
    return data.withColumn("level", normalize_level_udf("level"))
