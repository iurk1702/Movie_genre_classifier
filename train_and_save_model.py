"""
Train and save the movie genre classification model.
This script extracts the model training code from the notebook and saves the trained models.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, classification_report

from app.preprocessing import process_text

# Paths
BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "cornell-moviedialog-corpus"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Model paths
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
MULTILABEL_BINARIZER_PATH = MODELS_DIR / "multilabel_binarizer.pkl"
CLASSIFIER_PATH = MODELS_DIR / "classifier.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"

# 11 genres that cover 91% of the dataset
NEW_GENRE_LIST = [
    'drama', 'thriller', 'action', 'comedy', 'crime', 'romance',
    'sci-fi', 'adventure', 'mystery', 'horror', 'fantasy'
]


def load_data():
    """Load the Cornell movie dataset."""
    print("Loading dataset...")
    
    dataset_path = DATASET_PATH
    
    # Load data files
    movie_lines = pd.read_csv(
        f'{dataset_path}/movie_lines.txt',
        sep=r'\+\+\+\$\+\+\+',
        header=None,
        names=['Dialogue_ID', 'speaker_ID1', 'movie_ID', 'speaker', 'Dialogue'],
        encoding='unicode_escape',
        engine='python'
    )
    
    movie_titles_meta = pd.read_csv(
        f'{dataset_path}/movie_titles_metadata.txt',
        sep=r'\+\+\+\$\+\+\+',
        header=None,
        names=['movie_ID', 'movie_name', 'year', 'rating', 'no_of_votes', 'list_of_genres'],
        encoding='unicode_escape',
        engine='python'
    )
    
    # Clean movie IDs
    movie_lines['movie_ID'] = movie_lines['movie_ID'].apply(str.strip)
    movie_titles_meta['movie_ID'] = movie_titles_meta['movie_ID'].apply(str.strip)
    
    return movie_lines, movie_titles_meta


def prepare_data(movie_lines, movie_titles_meta):
    """Prepare data for training."""
    print("Preparing data...")
    
    # Filter movies that have at least one of the 11 genres
    matching_rows = movie_titles_meta[
        movie_titles_meta['list_of_genres'].apply(
            lambda x: any(i in x for i in NEW_GENRE_LIST)
        )
    ]
    movie_ID_select_genre = matching_rows['movie_ID'].unique()
    
    # Get genres for selected movies
    interim_genre11_df = movie_titles_meta[
        movie_titles_meta['list_of_genres'].apply(
            lambda x: any(i in x for i in NEW_GENRE_LIST)
        )
    ]
    genres_11_pandas_series = interim_genre11_df['list_of_genres']
    genres_11_list = list(genres_11_pandas_series)
    
    # Process genre lists
    genres_11_list_final = []
    for index in range(len(genres_11_list)):
        row = genres_11_list[index].replace('" ', "").replace('"', "").replace("'", '').replace("[", "").replace("]", "").strip()
        genres_11_list_final.append(row.split(", "))
    
    # Get dialogues for selected movies
    interim = movie_lines.loc[movie_lines['movie_ID'].isin(movie_ID_select_genre)].copy()
    result = interim['Dialogue'].apply(str)
    
    # Preprocess dialogues
    print("Preprocessing dialogues...")
    interim["Dialogue_processed"] = result.apply(process_text)
    
    # Clean up processed dialogues
    for index, value in interim['Dialogue_processed'].items():
        new_value = value.replace('"', "").replace("'", "").replace("``", "").strip()
        interim.at[index, 'Dialogue_processed'] = new_value
    
    # Create list of documents (one per movie)
    list_of_documents_11genres = []
    for movie_id in movie_ID_select_genre:
        selected_rows = interim[interim['movie_ID'] == movie_id]
        Dialogue_column_values = selected_rows['Dialogue_processed']
        Dialogue_column_values_list = Dialogue_column_values.tolist()
        Dialogue_column_values_list = [x for x in Dialogue_column_values_list if x != '']
        list_of_documents_11genres.append(' '.join([str(elem) for elem in Dialogue_column_values_list]))
    
    return list_of_documents_11genres, genres_11_list_final


def train_model(list_of_documents, genres_list):
    """Train the genre classification model."""
    print("Training model...")
    
    # Create vectorizer and transform documents
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(list_of_documents)
    
    # Create multilabel binarizer
    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit(genres_list)
    y = multilabel_binarizer.transform(genres_list)
    
    # Split data
    xtrain, xval, ytrain, yval = train_test_split(x, y, test_size=0.2, random_state=9)
    
    # Train classifier
    lr = LogisticRegression(max_iter=1000, penalty='l2')
    clf = OneVsRestClassifier(lr)
    clf.fit(xtrain, ytrain)
    
    # Evaluate
    y_pred = clf.predict(xval)
    f1 = f1_score(yval, y_pred, average="weighted", zero_division=0)
    
    print(f"\nModel trained successfully!")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"Number of training samples: {xtrain.shape[0]}")
    print(f"Number of validation samples: {xval.shape[0]}")
    print(f"Number of features: {x.shape[1]}")
    print(f"Number of genres: {len(multilabel_binarizer.classes_)}")
    print(f"Genres: {list(multilabel_binarizer.classes_)}")
    
    # Generate classification report
    report = classification_report(yval, y_pred, target_names=multilabel_binarizer.classes_, output_dict=True, zero_division=0)
    
    return vectorizer, multilabel_binarizer, clf, f1, report


def save_models(vectorizer, multilabel_binarizer, classifier, f1_score, classification_report):
    """Save trained models and metadata."""
    print("\nSaving models...")
    
    # Save models
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"✓ Saved vectorizer to {VECTORIZER_PATH}")
    
    joblib.dump(multilabel_binarizer, MULTILABEL_BINARIZER_PATH)
    print(f"✓ Saved multilabel binarizer to {MULTILABEL_BINARIZER_PATH}")
    
    joblib.dump(classifier, CLASSIFIER_PATH)
    print(f"✓ Saved classifier to {CLASSIFIER_PATH}")
    
    # Save metadata
    metadata = {
        "version": "1.0.0",
        "training_date": datetime.now().isoformat(),
        "genres": list(multilabel_binarizer.classes_),
        "num_genres": len(multilabel_binarizer.classes_),
        "f1_score": float(f1_score),
        "model_type": "OneVsRestClassifier with LogisticRegression",
        "preprocessing": "Tokenization, lemmatization, stopword removal",
        "features": "CountVectorized Bag of Words",
        "classification_report": classification_report
    }
    
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {METADATA_PATH}")


def main():
    """Main training function."""
    print("=" * 60)
    print("Movie Genre Classifier - Model Training")
    print("=" * 60)
    
    try:
        # Load data
        movie_lines, movie_titles_meta = load_data()
        
        # Prepare data
        list_of_documents, genres_list = prepare_data(movie_lines, movie_titles_meta)
        
        # Train model
        vectorizer, multilabel_binarizer, classifier, f1, report = train_model(
            list_of_documents, genres_list
        )
        
        # Save models
        save_models(vectorizer, multilabel_binarizer, classifier, f1, report)
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

