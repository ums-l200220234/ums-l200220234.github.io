import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import  Normalizer
from sklearn.feature_extraction.text import CountVectorizer
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import string
import nltk
import tarfile

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')


        
# factory = StemmerFactory()
# stemmer = factory.create_stemmer()
lemmatizer = WordNetLemmatizer()


def clean_data(df):
    df = df.dropna(subset=['Konten'])
    df = df[df['Konten'].str.strip() != ""]
    print(f"Dataset cleaned. Remaining records: {len(df)}.")
    return df

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    # stemmed_words = [stemmer.stem(word) for word in words]
    lemmatized_words = [lemmatizer.lemmatize(word, pos=wordnet.VERB) for word in words]
    return ' '.join(lemmatized_words)

def preprocess_dataset(df):
    df = clean_data(df)
    df['Processed_Konten'] = df['Konten'].apply(preprocess_text)
    print("Preprocessing completed.")
    return df

def scale_data(text_data, binary=False):
    vec = CountVectorizer(min_df=1, max_df=1,binary=binary)
    mtx = vec.fit_transform(text_data)
    cols = [None] * len(vec.vocabulary_)
    for word, idx in vec.vocabulary_.items():
        cols[idx] = word
    return mtx, cols

def load_chat(nums_docs):
    file_path = "../UAS/chat_cleaned.csv"
    try:
        df = pd.read_csv(file_path)
    except:
        print("file not found error")

    df = clean_data(df)
    df = preprocess_dataset(df)
    return df

        

