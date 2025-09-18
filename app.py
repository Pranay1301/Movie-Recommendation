import pickle
import streamlit as st
import requests
import pandas as pd
import numpy as np

# --- Configuration ---
TMDB_API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual TMDB API key

# --- Core Recommendation Logic ---

def get_hybrid_recommendations(movie, movies_df, similarity_matrix, alpha=0.7):
    """
    Recommends movies based on a hybrid score of similarity and weighted rating.
    alpha: weight for similarity score (e.g., 0.7 means 70% similarity, 30% quality).
    """
    try:
        # Find the movie in the dataframe
        movie_mask = movies_df['title'] == movie
        if not movie_mask.any():
            st.error("Movie not found in the dataset. Please select another one.")
            return [], [], [], []
        
        # Get the position index (not the DataFrame index)
        index = movie_mask.idxmax()
        
        # Convert DataFrame index to position index if necessary
        if hasattr(movies_df, 'reset_index'):
            # Find the position of this movie in the dataframe
            movie_position = movies_df.index.get_loc(index)
        else:
            movie_position = index
            
        # Ensure the index is within bounds
        if movie_position >= len(similarity_matrix):
            st.error(f"Index error: Movie index {movie_position} is out of bounds for similarity matrix with size {len(similarity_matrix)}")
            return [], [], [], []
            
    except Exception as e:
        st.error(f"Error finding movie: {str(e)}")
        return [], [], [], []

    # Get similarity scores for the selected movie
    try:
        sim_scores = list(enumerate(similarity_matrix[movie_position]))
    except IndexError as e:
        st.error(f"Similarity matrix index error: {str(e)}")
        return [], [], [], []

    # --- Hybrid Score Calculation ---
    hybrid_scores = []
    
    # Get min and max scores for normalization
    min_score = movies_df['score'].min()
    max_score = movies_df['score'].max()
    score_range = max_score - min_score
    
    for i, sim_score in sim_scores:
        # Ensure we don't go out of bounds
        if i >= len(movies_df):
            continue
            
        # Get the weighted rating ('score') for the movie at index i
        try:
            movie_score = movies_df.iloc[i]['score']
        except (IndexError, KeyError):
            continue
        
        # Normalize the movie score (0-1)
        if score_range > 0:
            normalized_score = (movie_score - min_score) / score_range
        else:
            normalized_score = 0.5  # Default if all scores are the same
        
        # Calculate hybrid score
        final_score = (alpha * sim_score) + ((1 - alpha) * normalized_score)
        hybrid_scores.append((i, final_score))

    # Sort movies based on hybrid score (descending)
    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)

    # Get top 5 movies (excluding the selected movie itself)
    top_movies = []
    for i, score in hybrid_scores[1:]:  # Skip the first one (selected movie)
        if len(top_movies) >= 5:
            break
        if i < len(movies_df):  # Ensure index is valid
            top_movies.append((i, score))

    # --- Prepare Output ---
    recommended_movie_names = []
    recommended_movie_posters = []
    recommended_movie_ratings = []
    recommended_movie_ids = []

    for i, score in top_movies:
        movie_info = movies_df.iloc[i]
        movie_id = movie_info['movie_id']
        
        recommended_movie_ids.append(movie_id)
        recommended_movie_names.append(movie_info['title'])
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_ratings.append(round(movie_info['score'], 2))

    return recommended_movie_names, recommended_movie_posters, recommended_movie_ratings, recommended_movie_ids

def fetch_poster(movie_id):
    """Fetches movie poster from TMDB API."""
    if TMDB_API_KEY == "YOUR_API_KEY_HERE":
        # Return placeholder if API key not set
        return "https://via.placeholder.com/500x750.png?text=No+API+Key"
    
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        response = requests.get(url, timeout=5)
        data = response.json()
        poster_path = data.get('poster_path', '')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
        else:
            return "https://via.placeholder.com/500x750.png?text=No+Poster"
    except Exception as e:
        return "https://via.placeholder.com/500x750.png?text=Error+Loading"

# --- Page Setup and App ---
st.set_page_config(
    page_title="Movie Recommender", 
    layout="wide",
    page_icon="🎬"
)

# Custom CSS (optional - will use default if file not found)
try:
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    local_css("style.css")
except FileNotFoundError:
    # Use default styling if CSS file not found
    st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stSelectbox > div > div > select {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

st.title('🎬 Enhanced Movie Recommendation System ✨')
st.markdown("Get personalized movie recommendations based on content similarity and quality scores!")

# Load the model files
try:
    with st.spinner('Loading recommendation model...'):
        movies_dict = pickle.load(open('movies_list_enhanced.pkl', 'rb'))
        similarity = pickle.load(open('similarity_enhanced.pkl', 'rb'))
        movies = pd.DataFrame(movies_dict)
        
    st.success(f"✅ Model loaded successfully! {len(movies)} movies available for recommendations.")
    
except FileNotFoundError as e:
    st.error("❌ Model files not found. Please run the model_builder.py script first to generate the required files.")
    st.info("Required files: movies_list_enhanced.pkl, similarity_enhanced.pkl")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model files: {str(e)}")
    st.stop()

# Create the movie selection interface
movie_list = sorted(movies['title'].values)
selected_movie = st.selectbox(
    "🎯 Select a movie you like to get high-quality recommendations:",
    movie_list,
    help="Choose a movie from our database to get similar recommendations"
)

# Add some information about the selected movie
if selected_movie:
    movie_info = movies[movies['title'] == selected_movie].iloc[0]
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"📊 **Selected Movie:** {selected_movie} | **Quality Score:** {round(movie_info['score'], 2)}")
    with col2:
        st.image(fetch_poster(movie_info['movie_id']), width=100)

# Recommendation button and results
if st.button('🚀 Get Recommendations', type='primary'):
    with st.spinner('🔍 Calculating recommendations with the enhanced model...'):
        names, posters, ratings, ids = get_hybrid_recommendations(selected_movie, movies, similarity)
        
        if names:
            st.success("🎉 Here are your personalized recommendations!")
            st.subheader("🏆 Top 5 Recommendations (Sorted by Similarity & Quality):")
            
            # Display recommendations in columns
            cols = st.columns(5)
            for i, col in enumerate(cols):
                with col:
                    st.markdown(f"**{names[i]}**")
                    st.image(posters[i], use_column_width=True)
                    st.markdown(f"⭐ **Rating: {ratings[i]}**")
                    
                    # Add some spacing
                    st.markdown("---")
        else:
            st.error("❌ Could not generate recommendations. Please try selecting a different movie.")

# Add footer information
st.markdown("---")
with st.expander("ℹ️ About this Recommendation System"):
    st.markdown("""
    **How it works:**
    - Uses content-based filtering on the TMDB 5000 dataset
    - Applies NLP techniques (TF-IDF, stemming) to analyze movie features
    - Considers genres, keywords, cast, crew, and plot overview
    - Combines similarity scores with quality ratings for better recommendations
    - Uses weighted ratings based on the IMDB formula
    
    **Features:**
    - Hybrid recommendation system (70% similarity + 30% quality)
    - Enhanced weighting for important features (director, genres, keywords)
    - High-quality movie filtering (90th percentile vote threshold)
    """)

if TMDB_API_KEY == "YOUR_API_KEY_HERE":
    st.warning("⚠️ To see movie posters, please add your TMDB API key in the app.py file")