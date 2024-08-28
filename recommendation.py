import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import os

# Load your dataset
recipe_data = pd.read_csv("merged_data.csv")

def preprocess_data(data):
    # Vectorize the ingredients list
    vectorizer = TfidfVectorizer(max_features=2000)
    X_ingredients = vectorizer.fit_transform(data['ingredients_list'])

    # Apply Truncated SVD to reduce dimensionality
    svd = TruncatedSVD(n_components=100)
    X_ingredients_reduced = svd.fit_transform(X_ingredients)

    # Scale the numerical features
    scaler = StandardScaler()
    X_numerical = scaler.fit_transform(data[['prep_time', 'calories', 'fat', 'carbohydrates', 'protein', 'cholesterol', 'sodium', 'fiber']])

    # Combine numerical and reduced ingredient features
    X_combined = np.hstack([X_numerical, X_ingredients_reduced])

    # Fit the KNN model
    knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
    knn.fit(X_combined)

    # Save the preprocessed components to cache
    joblib.dump((vectorizer, svd, scaler, knn), 'preprocessed_data.pkl')

    return vectorizer, svd, scaler, knn

def load_or_preprocess_data(data):
    # Check if the cache file exists
    if os.path.exists('preprocessed_data.pkl'):
        vectorizer, svd, scaler, knn = joblib.load('preprocessed_data.pkl')
    else:
        vectorizer, svd, scaler, knn = preprocess_data(data)

    return vectorizer, svd, scaler, knn

def recommend_recipes(input_features, vectorizer, svd, scaler, knn, data):
    # Scale and transform the input features
    input_features_scaled = scaler.transform([input_features[:8]])
    input_ingredients_transformed = vectorizer.transform([input_features[8]])
    input_ingredients_reduced = svd.transform(input_ingredients_transformed)
    input_combined = np.hstack([input_features_scaled, input_ingredients_reduced])

    # Find the nearest neighbors
    distances, indices = knn.kneighbors(input_combined)
    
    # Get the recommended recipes
    recommendations = data.iloc[indices[0]]
    return recommendations[['recipe_name', 'prep_time', 'ingredients_list']].head(5)

# Load or preprocess the data
vectorizer, svd, scaler, knn = load_or_preprocess_data(recipe_data)

# Example input features for recommendation
input_features = [30, 200, 10, 30, 20, 50, 300, 5, "chicken, garlic, onion, tomato, pepper"]

# Get recommendations
recommendations = recommend_recipes(input_features, vectorizer, svd, scaler, knn, recipe_data)

# Display the recommendations
print(recommendations)
