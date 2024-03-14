import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('stopwords')
nltk.download('punkt')
from nltk import word_tokenize
from nltk.corpus import stopwords


film_data_processed = pd.read_csv('film_data_processed.csv')
df = pd.read_csv('data_ass2_part2_wk5.csv')

tfidf = TfidfVectorizer(min_df = 2, max_df = 0.8)

bag_of_words = tfidf.fit_transform(film_data_processed['bag_of_words'])

tfidf_df = pd.DataFrame(bag_of_words.toarray(), columns = tfidf.get_feature_names_out())
tfidf_df.index = df['title']
tfidf_df.to_csv("tfidf_data.csv")