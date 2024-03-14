import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk import word_tokenize
from nltk.corpus import stopwords
import pickle
import pandas as pd
import streamlit as st
from streamlit import session_state as session

def recommend_movies(movies_liked, tfidf_matrix, recommendation_count=10):

    liked_movies_df = tfidf_matrix.reindex(movies_liked)
    user_profile = liked_movies_df.mean()
    
    recommendation_pool = tfidf_matrix.drop(movies_liked)
    
    similarity_scores = cosine_similarity(user_profile.values.reshape(1, -1), recommendation_pool)
    similarity_scores_df = pd.DataFrame(similarity_scores.T, index=recommendation_pool.index, columns=["similarity_score"])
    
    recommended_movies_df = similarity_scores_df.sort_values(by="similarity_score", ascending=False).head(recommendation_count)

    return recommended_movies_df

tfidf = pd.read_csv("tfidf_data.csv", index_col=0)

with open("movie_titles.pickle", "rb") as f:
    movies = pickle.load(f)


dataframe = None

st.title("""
Content-Based Filtering for Week 5 dataset 
For Marketing Analytics (6013B0806Y) made by L Chen.
 """)

st.text("")
st.text("")
st.text("")
st.text("")

session.options = st.multiselect(label="Movie Titles", options=movies)

st.text("")
st.text("")

session.slider_count = st.slider(label="Number of recommended movies", min_value=10, max_value=50)

st.text("")
st.text("")

buffer1, col1, buffer2 = st.columns([1.6, 1, 1])

is_clicked = col1.button(label="One click to recommend")

if is_clicked:
    dataframe = recommend_movies(session.options, recommendation_count=session.slider_count, tfidf_matrix=tfidf)

st.text("")
st.text("")
st.text("")
st.text("")


if dataframe is not None:
    st.table(dataframe)