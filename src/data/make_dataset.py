# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os

from pyspark.sql.types import StringType, StructField, StructType

from src.data.clean_data import clean_data
from pyspark.sql import SparkSession
import os

from src.data.data_unifier import unify_dataframes


def load_processed():
    # Initialisation de SparkSession
    spark = SparkSession.builder.appName("RecommandationCours").getOrCreate()
    schema = StructType([
        StructField("title", StringType(), nullable=True),
        StructField("description", StringType(), nullable=True),
        StructField("level", StringType(), nullable=True),
        StructField("platform", StringType(), nullable=True)
   ])
    dataframes = spark.createDataFrame([],schema=schema)
    for file in os.listdir('data/processed'):
        if file.endswith('.csv'):
            # Lecture du fichier CSV en DataFrame
            file_path = os.path.join('data/processed', file)
            df = spark.read.csv(file_path, header=True, inferSchema=True, escape='"')
            dataframes = dataframes.union(df)

    return dataframes


def load_data(repo_path):
    # Initialisation de SparkSession
    spark = SparkSession.builder.appName("RecommandationCours").getOrCreate()

    dataframes = {}
    for file in os.listdir(repo_path):
        # Construction du chemin complet du fichier
        file_path = os.path.join(repo_path, file)

        # Vérification si c'est un fichier et non un dossier
        if os.path.isfile(file_path):
            # Lecture du fichier CSV en DataFrame
            df = spark.read.csv(file_path, header=True, inferSchema=True, multiLine=True, escape='"')
            # Stockage du DataFrame dans le dictionnaire avec le nom du fichier comme clé
            filename = os.path.splitext(file)[0]
            dataframes[filename] = df

    return dataframes



def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    raw_data_path = os.path.join(project_dir, 'data/raw/courses')
    processed_data_path = os.path.join(project_dir, 'data/processed')
    dataframes = load_data(raw_data_path)

    coursera_df = dataframes["coursera"]
    edx_df = dataframes["edx"]
    udemy_df = dataframes["udemy"]

    # Unifier les DataFrames
    common_columns = ['title', 'description', 'level', 'platform']  # ajustez selon vos données

    unified_df = unify_dataframes(coursera_df, edx_df, udemy_df, common_columns)

    df = clean_data(unified_df)
    df.write.csv(processed_data_path, header=True, mode="overwrite")
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
