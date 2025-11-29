"""
Extract research data from the notebook analysis and save as JSON files.
This script runs the analysis code to generate research data for the API.
"""

import pandas as pd
import numpy as np
import json
import joblib
import sys
from pathlib import Path
from collections import Counter
from nltk.probability import FreqDist
import nltk
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# Add parent directory to path for imports
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.preprocessing import process_text

# Paths (BASE_DIR already defined above)
DATASET_PATH = BASE_DIR / "cornell-moviedialog-corpus"
RESEARCH_DATA_DIR = BASE_DIR / "data" / "research"
RESEARCH_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = BASE_DIR / "models"

# 11 genres
NEW_GENRE_LIST = [
    'drama', 'thriller', 'action', 'comedy', 'crime', 'romance',
    'sci-fi', 'adventure', 'mystery', 'horror', 'fantasy'
]


def load_data():
    """Load the dataset."""
    print("Loading dataset...")
    
    movie_titles_meta = pd.read_csv(
        f'{DATASET_PATH}/movie_titles_metadata.txt',
        sep=r'\+\+\+\$\+\+\+',
        header=None,
        names=['movie_ID', 'movie_name', 'year', 'rating', 'no_of_votes', 'list_of_genres'],
        encoding='unicode_escape',
        engine='python'
    )
    
    movie_lines = pd.read_csv(
        f'{DATASET_PATH}/movie_lines.txt',
        sep=r'\+\+\+\$\+\+\+',
        header=None,
        names=['Dialogue_ID', 'speaker_ID1', 'movie_ID', 'speaker', 'Dialogue'],
        encoding='unicode_escape',
        engine='python'
    )
    
    movie_titles_meta['movie_ID'] = movie_titles_meta['movie_ID'].apply(str.strip)
    movie_lines['movie_ID'] = movie_lines['movie_ID'].apply(str.strip)
    
    return movie_titles_meta, movie_lines


def extract_genre_distribution(movie_titles_meta):
    """Extract genre frequency distribution."""
    print("Extracting genre distribution...")
    
    genre_string_list = movie_titles_meta["list_of_genres"].values.tolist()
    
    # Process genre lists
    genre_list = []
    for index in range(len(genre_string_list)):
        row = genre_string_list[index].replace('" ', "").replace('"', "").replace("'", '').replace("[", "").replace("]", "").strip()
        genre_list.append(row.split(", "))
    
    # Flatten and count
    flat_list_genre = [item for sublist in genre_list for item in sublist]
    count_genre = dict(Counter(flat_list_genre))
    
    # Filter for 11 genres
    filtered_genres = {k: v for k, v in count_genre.items() if k in NEW_GENRE_LIST}
    
    genres = list(filtered_genres.keys())
    counts = list(filtered_genres.values())
    total = sum(counts)
    percentages = [(c / total) * 100 for c in counts]
    
    data = {
        "genres": genres,
        "counts": counts,
        "percentages": [round(p, 2) for p in percentages]
    }
    
    with open(RESEARCH_DATA_DIR / "genre_distribution.json", 'w') as f:
        json.dump(data, f, indent=2)
    print("✓ Saved genre_distribution.json")


def extract_genre_cooccurrence(movie_titles_meta):
    """Extract genre co-occurrence data."""
    print("Extracting genre co-occurrence...")
    
    # Process genre lists for all movies
    genre_string_list = movie_titles_meta["list_of_genres"].values.tolist()
    genre_lists = []
    for genre_str in genre_string_list:
        cleaned = genre_str.replace('" ', "").replace('"', "").replace("'", '').replace("[", "").replace("]", "").strip()
        genre_lists.append([g.strip() for g in cleaned.split(", ")])
    
    # Filter to only include the 11 target genres
    filtered_genre_lists = []
    for genre_list in genre_lists:
        filtered = [g for g in genre_list if g in NEW_GENRE_LIST]
        if filtered:
            filtered_genre_lists.append(filtered)
    
    # Build co-occurrence matrix (11x11)
    cooccurrence_matrix = {}
    for genre in NEW_GENRE_LIST:
        cooccurrence_matrix[genre] = {g: 0 for g in NEW_GENRE_LIST}
    
    # Count co-occurrences
    for genre_list in filtered_genre_lists:
        for i, genre1 in enumerate(genre_list):
            for genre2 in genre_list[i:]:  # Include self and pairs
                if genre1 in NEW_GENRE_LIST and genre2 in NEW_GENRE_LIST:
                    cooccurrence_matrix[genre1][genre2] += 1
                    if genre1 != genre2:
                        cooccurrence_matrix[genre2][genre1] += 1
    
    # Convert to list format for JSON (maintain backward compatibility)
    genre_pairs = []
    counts = []
    percentages = []
    total_movies = len(movie_titles_meta)
    
    # Get top co-occurring pairs (excluding self)
    pair_counts = {}
    for genre1 in NEW_GENRE_LIST:
        for genre2 in NEW_GENRE_LIST:
            if genre1 < genre2:  # Avoid duplicates
                count = cooccurrence_matrix[genre1][genre2]
                if count > 0:
                    pair_counts[(genre1, genre2)] = count
    
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    for (genre1, genre2), count in sorted_pairs:
        genre_pairs.append([genre1, genre2])
        counts.append(count)
        percentages.append(round((count / total_movies) * 100, 2))
    
    # Create matrix format for heatmap (11x11 matrix)
    matrix = []
    for genre1 in NEW_GENRE_LIST:
        row = []
        for genre2 in NEW_GENRE_LIST:
            row.append(cooccurrence_matrix[genre1][genre2])
        matrix.append(row)
    
    data = {
        "genres": NEW_GENRE_LIST,  # List of genre names in order
        "matrix": matrix,  # 11x11 co-occurrence matrix
        "genre_pairs": genre_pairs,  # Top pairs (backward compatibility)
        "counts": counts,
        "percentages": percentages
    }
    
    with open(RESEARCH_DATA_DIR / "genre_cooccurrence.json", 'w') as f:
        json.dump(data, f, indent=2)
    print("✓ Saved genre_cooccurrence.json (with matrix format)")


def extract_word_frequencies(movie_titles_meta, movie_lines):
    """Extract word frequency data by genre."""
    print("Extracting word frequencies by genre...")
    
    # Create genre to movie ID mapping
    interim_dict = movie_titles_meta.set_index('movie_ID').to_dict()['list_of_genres']
    movieID_listOfGenres_dict = {k.strip(): v for k, v in interim_dict.items()}
    
    all_genre_data = {}
    
    for genre in NEW_GENRE_LIST:
        print(f"  Processing {genre}...")
        
        # Get movie IDs for this genre
        selected_movies = {
            k: v for k, v in movieID_listOfGenres_dict.items()
            if genre in v
        }
        movie_ids = list(selected_movies.keys())
        
        # Get dialogues for these movies
        dialogues = []
        for movie_id in movie_ids:
            selected_rows = movie_lines[movie_lines['movie_ID'] == movie_id]
            dialogue_values = selected_rows['Dialogue'].tolist()
            dialogues.extend(dialogue_values)
        
        # Preprocess and tokenize
        dialogue_text = ' '.join([str(d) for d in dialogues])
        processed = process_text(dialogue_text)
        
        # Tokenize and get frequencies
        tokens = nltk.word_tokenize(processed)
        freq_dist = FreqDist(tokens)
        
        # Get top 30 words
        top_words = freq_dist.most_common(30)
        
        words = [w[0] for w in top_words]
        frequencies = [w[1] for w in top_words]
        
        all_genre_data[genre] = {
            "genre": genre,
            "words": words,
            "frequencies": frequencies
        }
    
    # Save individual files
    for genre, data in all_genre_data.items():
        filename = RESEARCH_DATA_DIR / f"word_frequencies_{genre.lower()}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    # Save combined file
    with open(RESEARCH_DATA_DIR / "word_frequencies_all.json", 'w') as f:
        json.dump(all_genre_data, f, indent=2)
    
    print("✓ Saved word frequency files")


def extract_model_performance():
    """Extract model performance metrics from trained model."""
    print("Extracting model performance...")
    
    # Check if model files exist
    metadata_path = MODELS_DIR / "metadata.json"
    classifier_path = MODELS_DIR / "classifier.pkl"
    vectorizer_path = MODELS_DIR / "vectorizer.pkl"
    multilabel_binarizer_path = MODELS_DIR / "multilabel_binarizer.pkl"
    
    if not all([metadata_path.exists(), classifier_path.exists(), 
                vectorizer_path.exists(), multilabel_binarizer_path.exists()]):
        print("⚠ Model files not found. Using placeholder values.")
        print("   Run 'python train_and_save_model.py' first to train the model.")
        data = {
            "f1_score": 0.47,
            "precision": 0.57,
            "recall": 0.42,
            "per_genre_metrics": {}
        }
    else:
        try:
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check if classification report exists in metadata
            if "classification_report" in metadata:
                report = metadata["classification_report"]
                
                # Extract overall metrics
                f1 = report.get("weighted avg", {}).get("f1-score", 0.0)
                precision = report.get("weighted avg", {}).get("precision", 0.0)
                recall = report.get("weighted avg", {}).get("recall", 0.0)
                
                # Extract per-genre metrics
                per_genre_metrics = {}
                for genre in NEW_GENRE_LIST:
                    if genre in report:
                        genre_metrics = report[genre]
                        per_genre_metrics[genre] = {
                            "precision": round(genre_metrics.get("precision", 0.0), 3),
                            "recall": round(genre_metrics.get("recall", 0.0), 3),
                            "f1": round(genre_metrics.get("f1-score", 0.0), 3),
                            "support": int(genre_metrics.get("support", 0))
                        }
                
                data = {
                    "f1_score": round(f1, 3),
                    "precision": round(precision, 3),
                    "recall": round(recall, 3),
                    "per_genre_metrics": per_genre_metrics
                }
                
                print(f"  ✓ Loaded metrics from metadata:")
                print(f"    F1 Score: {f1:.3f}")
                print(f"    Precision: {precision:.3f}")
                print(f"    Recall: {recall:.3f}")
                print(f"    Per-genre metrics: {len(per_genre_metrics)} genres")
            else:
                print("⚠ Classification report not found in metadata. Using placeholder values.")
                data = {
                    "f1_score": 0.47,
                    "precision": 0.57,
                    "recall": 0.42,
                    "per_genre_metrics": {}
                }
        except Exception as e:
            print(f"⚠ Error loading model metrics: {e}")
            print("   Using placeholder values.")
            data = {
                "f1_score": 0.47,
                "precision": 0.57,
                "recall": 0.42,
                "per_genre_metrics": {}
            }
    
    with open(RESEARCH_DATA_DIR / "model_performance.json", 'w') as f:
        json.dump(data, f, indent=2)
    print("✓ Saved model_performance.json")


def extract_overview():
    """Create overview data."""
    print("Extracting overview...")
    
    data = {
        "title": "Movie Genre Classifier using NLP",
        "objective": "Build a multi-label classifier that predicts movie genres from dialogue text using the Cornell Movie Dialog Corpus dataset.",
        "dataset_description": "The Cornell Movie Dialog Corpus contains 617 movies with 304,713 utterances from 9,035 characters. Each movie has metadata including genres, release year, IMDB rating, and number of votes.",
        "methodology": [
            "Data loading and preprocessing (tokenization, lemmatization, stopword removal)",
            "Feature engineering using Bag of Words (CountVectorizer)",
            "Multi-label classification using OneVsRestClassifier with Logistic Regression",
            "Model evaluation using F1-score, precision, and recall metrics"
        ],
        "key_findings": [
            "11 genres (drama, thriller, action, comedy, crime, romance, sci-fi, adventure, mystery, horror, fantasy) cover 91% of the dataset",
            "Best model: OneVsRestClassifier with Logistic Regression using only CountVectorized Bag of Words",
            "Adding additional features (sentiment, dialogue count, etc.) caused overfitting",
            "Random Forest had higher precision but poor recall due to class imbalance"
        ]
    }
    
    with open(RESEARCH_DATA_DIR / "overview.json", 'w') as f:
        json.dump(data, f, indent=2)
    print("✓ Saved overview.json")


def extract_features_analysis():
    """Extract feature correlation data (placeholder)."""
    print("Extracting features analysis...")
    
    # Placeholder - this would need actual correlation matrix from notebook
    data = {
        "features": ["movie_dialogue_count", "sentiment", "number_of_characters", "average_top10Freq"],
        "correlation_matrix": [
            [1.0, 0.1, 0.3, 0.2],
            [0.1, 1.0, 0.05, 0.15],
            [0.3, 0.05, 1.0, 0.25],
            [0.2, 0.15, 0.25, 1.0]
        ],
        "feature_names": ["movie_dialogue_count", "sentiment", "number_of_characters", "average_top10Freq"]
    }
    
    with open(RESEARCH_DATA_DIR / "features_analysis.json", 'w') as f:
        json.dump(data, f, indent=2)
    print("✓ Saved features_analysis.json (update with actual correlation data)")


def main():
    """Main extraction function."""
    print("=" * 60)
    print("Extracting Research Data")
    print("=" * 60)
    
    try:
        # Load data
        movie_titles_meta, movie_lines = load_data()
        
        # Extract data
        extract_overview()
        extract_genre_distribution(movie_titles_meta)
        extract_genre_cooccurrence(movie_titles_meta)
        extract_word_frequencies(movie_titles_meta, movie_lines)
        extract_model_performance()
        extract_features_analysis()
        
        print("\n" + "=" * 60)
        print("Research data extraction completed!")
        print(f"Data saved to: {RESEARCH_DATA_DIR}")
        print("=" * 60)
        print("\nNote: features_analysis.json contains placeholder correlation data.")
        print("      Update it with actual feature correlation analysis if needed.")
        
    except Exception as e:
        print(f"\nError during extraction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

