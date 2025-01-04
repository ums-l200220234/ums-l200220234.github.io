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
    df['Konten'] = df['Konten'].apply(preprocess_text)
    print("Preprocessing completed.")
    return df

def scale_data(text_data, binary=False):
    processed = text_data['Konten']
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(processed)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    normalizer = Normalizer()
    mtx = normalizer.fit_transform(tfidf_matrix)
    print(f"Normalized matrix shape: {mtx.shape}")

    cols = vectorizer.get_feature_names_out()
    return mtx, cols

def load_chat(nums_docs):
    tar_path =  "../UAS/chat_cleaned.tar"
    file_name = "chat_cleaned.csv"
    try:
        with tarfile.open(tar_path, "r:*") as tar: 
            member = tar.getmember(file_name)
            with tar.extractfile(member) as f:
                df = pd.read_csv(f) 
                df = clean_data(df)
                df = preprocess_dataset(df)
        print(f"File '{file_name}' successfully loaded from '{tar_path}'.")
        return df

    except FileNotFoundError:
        print(f"File '{tar_path}' not found.")
    except KeyError:
        print(f"'{file_name}' not found in the archive.")
    except Exception as e:
        print(f"Error occurred: {e}")


    return df


        

