import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import  Normalizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import numpy as np
import string
import nltk
import re
import tarfile

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')


        
factory = StemmerFactory()
stemmer = factory.create_stemmer()
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
    stemmed_words = [stemmer.stem(word) for word in words]
    lemmatized_words = [lemmatizer.lemmatize(word, pos=wordnet.VERB) for word in stemmed_words]
    return ' '.join(lemmatized_words)

def preprocess_dataset(df):
    df = clean_data(df)
    df['Processed_Konten'] = df['Konten'].apply(preprocess_text)
    print("Preprocessing completed.")
    return df

def scale_data(text_data):
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')  
    tfidf_matrix = vectorizer.fit_transform(text_data)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    normalizer = Normalizer()
    scaled_matrix = normalizer.fit_transform(tfidf_matrix)
    print(f"Normalized matrix shape: {scaled_matrix.shape}")
    
    return scaled_matrix, vectorizer.get_feature_names_out()

def load_tar_data(tar_path, file_name):
    with tarfile.open(tar_path, 'r') as tar:
        extracted_file = tar.extractfile(file_name)
        if extracted_file is not None:
            df = pd.read_csv(extracted_file)
            df = clean_data(df)
            df = preprocess_dataset(df)
            return df
        else:
            raise FileNotFoundError(f"File {file_name} tidak ditemukan dalam arsip.")