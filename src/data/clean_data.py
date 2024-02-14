import nltk as nltk
from nltk.corpus import stopwords
import re
import string

from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType

nltk.download('stopwords')
stop = set(stopwords.words('english'))


def clean_text(text):
    removed=["Learn" ,"master", "course" ,"using", "guide" ,"create" ,"skills" ,"guide" ,"complete" ,"create"]
    text=str(text).lower() #Converts text to lowercase
    text=re.sub('\d+', '', text) #removes numbers
    text=re.sub('\[.*?\],', '', text) #removes HTML tags
    text=re.sub('https?://\S+|www\.\S+', '', text) #removes url
    text=re.sub(r"["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", "", text) #removes emojis
    text=re.sub('[%s]' % re.escape(string.punctuation),'',text) #removes punctuations
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = ' '.join(word for word in text.split() if word not in stop)  # Supprime les stopwords
    text = ' '.join(word for word in text.split() if word not in removed)  # Supprime les stopwords
    return text

clean_text_udf = udf(clean_text, StringType())
def clean_data(data):
    data= data.na.drop()

    cleaned_df = data.withColumn("description", clean_text_udf(col("description")))
    cleaned_df = cleaned_df.withColumn("title", clean_text_udf(col("title")))
    cleaned_df = cleaned_df.filter(data["description"].isNotNull())
    return cleaned_df

def remove_words(text, words_to_remove):
    # Diviser le texte en mots
    words = text.split()

    # Filtrer les mots à supprimer
    filtered_words = [word for word in words if word.lower() not in words_to_remove]

    # Recombiner en une chaîne de texte
    return ' '.join(filtered_words)
