import numpy as np
import pandas as pd
import ast
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# --- 1. DATA LOADING AND PREPROCESSING ---
print("Loading datasets...")
movies_raw = pd.read_csv('tmdb_5000_movies.csv')
credits_raw = pd.read_csv('tmdb_5000_credits.csv')

# Merge datasets on title
movies = movies_raw.merge(credits_raw, on='title')

# Select relevant columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'vote_average', 'vote_count']]

# Drop rows with missing values
movies.dropna(inplace=True)
print(f"Dataset loaded with {len(movies)} movies")

# --- 2. CALCULATE WEIGHTED RATING (IMDB Formula) ---
print("Calculating weighted ratings...")

# Calculate the mean vote across all movies (C)
C = movies['vote_average'].mean()

# Calculate the minimum number of votes required (90th percentile) (m)
m = movies['vote_count'].quantile(0.9)

# Filter movies that meet the minimum vote count threshold
q_movies = movies.copy().loc[movies['vote_count'] >= m]

# Function to calculate weighted rating
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (v + m) * C)

# Add the 'score' column
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

# Sort movies by score
q_movies = q_movies.sort_values('score', ascending=False)

# Use qualified movies for our model
movies = q_movies.copy()
print(f"Using {len(movies)} qualified movies")

# --- 3. HELPER FUNCTIONS FOR DATA TRANSFORMATION ---

def convert(text):
    """Convert JSON-like strings to list of names"""
    L = []
    try:
        for i in ast.literal_eval(text):
            L.append(i['name'])
    except:
        pass
    return L

def convert_cast(text):
    """Get top 3 cast members"""
    L = []
    counter = 0
    try:
        for i in ast.literal_eval(text):
            if counter < 3:
                L.append(i['name'])
                counter += 1
    except:
        pass
    return L

def fetch_director(text):
    """Fetch director name from crew"""
    L = []
    try:
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
    except:
        pass
    return L

def collapse(L):
    """Remove spaces between names to create single tags"""
    L1 = []
    for i in L:
        L1.append(i.replace(" ", ""))
    return L1

def process_genres(obj):
    """Process and weight genres"""
    return collapse(convert(obj)) * 2  # Weight genres more

def process_keywords(obj):
    """Process and weight keywords"""
    return collapse(convert(obj)) * 2  # Weight keywords more

def process_director(obj):
    """Process and weight director"""
    return collapse(fetch_director(obj)) * 3  # Weight director most

# --- 4. APPLYING TRANSFORMATIONS ---
print("Processing movie features...")

# Apply transformations with weighting
movies['genres'] = movies['genres'].apply(process_genres)
movies['keywords'] = movies['keywords'].apply(process_keywords)
movies['cast'] = movies['cast'].apply(convert_cast).apply(collapse)
movies['crew'] = movies['crew'].apply(process_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split() if pd.notna(x) else [])

# --- 5. CREATING THE 'TAGS' COLUMN ---
print("Creating tags...")

# Combine all processed features into 'tags'
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Create final dataframe
new_df = movies[['movie_id', 'title', 'tags', 'score']].copy()

# Convert tags to lowercase string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

# --- 6. TEXT VECTORIZATION & STEMMING ---
print("Applying stemming and vectorization...")

# Initialize Porter Stemmer
ps = PorterStemmer()

def stem(text):
    """Apply stemming to text"""
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

# Apply stemming
new_df['tags'] = new_df['tags'].apply(stem)

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')

# Transform tags into vectors
print("Computing TF-IDF vectors...")
vectors = tfidf.fit_transform(new_df['tags']).toarray()

# --- 7. COSINE SIMILARITY CALCULATION ---
print("Computing cosine similarity matrix...")
similarity = cosine_similarity(vectors)

# --- 8. RESET INDEX TO AVOID INDEX MISMATCH ---
print("Resetting dataframe index...")

# Reset index to ensure continuous indices from 0 to n-1
new_df = new_df.reset_index(drop=True)

# --- 9. SAVING THE MODEL AND DATA ---
print("Saving model files...")

# Save with the correct filenames that the app expects
pickle.dump(new_df.to_dict(), open('movies_list_enhanced.pkl', 'wb'))
pickle.dump(similarity, open('similarity_enhanced.pkl', 'wb'))

print("Model and data saved successfully!")
print(f"Final dataset shape: {new_df.shape}")
print(f"Similarity matrix shape: {similarity.shape}")
print("Files saved:")
print("- movies_list_enhanced.pkl")
print("- similarity_enhanced.pkl")