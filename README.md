# Enhanced Movie Recommendation System ✨

![Movie Recommender Banner](https://via.placeholder.com/1200x300?text=Enhanced+Movie+Recommendation+System+%F0%9F%8E%AC)
*(Placeholder banner; replace with a custom image for better visuals)*

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0-red?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Stars](https://img.shields.io/github/stars/pranaybhardwaj10/movie-recommender?style=social)](https://github.com/pranaybhardwaj10/movie-recommender)

---

## 📖 Project Overview

This is an **Enhanced Movie Recommendation System** built using content-based filtering on the TMDB 5000 Movies dataset. It combines NLP techniques (TF-IDF vectorization, stemming) with cosine similarity to recommend movies based on genres, keywords, cast, crew, and plot overviews. The system enhances recommendations by incorporating weighted quality scores (using IMDB's formula) and hybrid scoring (70% similarity + 30% quality).

Key technologies:
- **Python Libraries**: Pandas, NumPy, Scikit-learn, NLTK, Streamlit
- **Dataset**: TMDB 5000 Movies & Credits (4806 movies, filtered to 481 qualified ones)
- **Model**: TF-IDF + Cosine Similarity
- **Deployment**: Streamlit web app for interactive recommendations

The system filters for high-quality movies (90th percentile vote count) and weights important features like directors (x3), genres/keywords (x2) for more accurate suggestions.

---

## 🚀 Features

- **Hybrid Recommendations**: Blends content similarity with IMDB-weighted ratings.
- **Interactive UI**: Select a movie and get top 5 recommendations with posters, ratings, and details.
- **Data Processing**: Stemming, tokenization, and feature weighting for better accuracy.
- **TMDB Integration**: Fetches movie posters via API (requires your TMDB API key).
- **Model Building Script**: `model_builder.py` for generating pickled models.
- **Error Handling**: Graceful handling for missing movies or API issues.

---

## 📊 Visual Data & Insights

### Top 10 Movies by Weighted Score (from Dataset)
Here's a bar chart representing the top 10 movies based on IMDB-weighted scores (calculated in the notebook). Data extracted from the project's output:

| Movie Title                                   | Weighted Score |
|-----------------------------------------------|----------------|
| The Shawshank Redemption                      | 8.06           |
| Fight Club                                    | 7.94           |
| The Dark Knight                               | 7.92           |
| Pulp Fiction                                  | 7.90           |
| Inception                                     | 7.86           |
| The Godfather                                 | 7.85           |
| Interstellar                                  | 7.81           |
| Forrest Gump                                  | 7.80           |
| The Lord of the Rings: The Return of the King | 7.73           |
| The Empire Strikes Back                       | 7.70           |

#### Bar Graph (ASCII Art Representation)


*(For a real bar graph, embed an image generated via Matplotlib in your repo: e.g., save as `top_movies_bar.png`)*

![Top Movies Bar Graph](https://via.placeholder.com/800x400?text=Top+10+Movies+Bar+Graph+%28Replace+with+actual+image%29)

### Sample Similarity Distribution
From the cosine similarity matrix (shape: 481x481), here's a histogram-like view of similarity scores for a sample movie (e.g., The Shawshank Redemption):

- Mean Similarity: ~0.02
- Max Similarity (excluding self): ~0.35
- Distribution (binned):

*(Generate via code: `import matplotlib.pyplot as plt; plt.hist(similarity[0]); plt.savefig('similarity_hist.png')`)*

![Similarity Histogram](https://via.placeholder.com/600x300?text=Similarity+Score+Histogram+%28Replace+with+actual+image%29)

---

## 🛠 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/pranaybhardwaj10/movie-recommender.git](https://github.com/pranaybhardwaj10/movie-recommender.git)
    cd movie-recommender
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Create `requirements.txt` with: pandas, numpy, scikit-learn, nltk, streamlit, requests)*

3.  **Get a TMDB API Key:**
    -   Sign up at [TMDB](https://www.themoviedb.org/).
    -   Replace `YOUR_API_KEY_HERE` in `app.py`.

4.  **Build the model:**
    ```bash
    python model_builder.py
    ```
    This generates `movies_list_enhanced.pkl` and `similarity_enhanced.pkl`.

---

## 📱 Usage

1.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

2.  **Open in browser** (localhost:8501).
3.  **Select a movie** (e.g., "Spider-Man 2") and click "Get Recommendations".

### Screenshot
![App Screenshot](screenshot.png)
*(Screenshot of the app interface showing recommendations for Spider-Man 2: Spider-Man 3 (5.97), Spider-Man (6.62), The Dark Knight (7.92), Fight Club (7.94), The Shawshank Redemption (8.06))*

---

## 🔍 How It Works

1.  **Data Prep**: Load TMDB data, filter high-quality movies, compute weighted scores.
2.  **Feature Engineering**: Process genres/keywords/cast/crew with weighting and stemming.
3.  **Vectorization**: TF-IDF on combined tags.
4.  **Similarity**: Cosine similarity matrix.
5.  **Hybrid Scoring**: `hybrid_score = (0.7 * similarity) + (0.3 * normalized_quality)`.
6.  **App**: Streamlit for UI, TMDB for posters.

---

## 🤝 Contributing

1.  Fork the repo.
2.  Create a branch: `git checkout -b feature/new-feature`.
3.  Commit changes: `git commit -m "Add new feature"`.
4.  Push: `git push origin feature/new-feature`.
5.  Open a Pull Request.

Issues and suggestions welcome!

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Built by Pranay Bhardwaj. For questions, reach out at pranaybhardwaj10@gmail.com.
*Last updated: September 18, 2025*
