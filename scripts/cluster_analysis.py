# cluster_analysis.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# --- Global Constants ---
# File path for the dataset containing songs by single writers.
SINGLES_FILEPATH = "../srcs/single_writers_list.csv"


def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads song data from a CSV file.

    Args:
        filepath: The path to the CSV file.

    Returns:
        A pandas DataFrame containing the song data.
    """
    print("Loading data...")
    return pd.read_csv(filepath)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the DataFrame by cleaning the 'writers' column.

    Args:
        df: The input DataFrame.

    Returns:
        The preprocessed DataFrame.
    """
    print("Preprocessing writer data...")
    # Split the 'writers' string into a list of cleaned writer names.
    df['writers'] = df['writers'].apply(
        lambda x: [writer.strip() for writer in str(x).split(',')])
    return df


def create_features(df: pd.DataFrame) -> np.ndarray:
    """
    Generates combined features from text (TF-IDF) and sentiment scores.

    Args:
        df: The DataFrame containing song data.

    Returns:
        A NumPy array of combined features for clustering.
    """
    print("Creating feature vectors...")
    # 1. Text Vectorization using TF-IDF
    # Converts song lyrics into numerical vectors. Limited to the top 1000 features.
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])

    # 2. Sentiment and Year Feature Scaling
    # Normalize sentiment scores ('pos', 'neu', 'neg', 'compound') and 'year'.
    scaler = StandardScaler()
    sentiment_features = scaler.fit_transform(
        df[['pos', 'neu', 'neg', 'compound', 'year']])

    # 3. Combine TF-IDF features with scaled sentiment features.
    combined_features = np.hstack((tfidf_matrix.toarray(), sentiment_features))
    return combined_features


def perform_clustering(features: np.ndarray, num_clusters: int) -> np.ndarray:
    """
    Performs K-Means clustering on the given features.

    Args:
        features: The combined feature matrix.
        num_clusters: The desired number of clusters.

    Returns:
        An array of cluster labels for each song.
    """
    print(f"Performing K-Means clustering with {num_clusters} clusters...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(features)


def reduce_dimensionality(features: np.ndarray) -> np.ndarray:
    """
    Reduces feature dimensionality to 2D using PCA for visualization.

    Args:
        features: The high-dimensional feature matrix.

    Returns:
        A NumPy array with 2 principal components.
    """
    print("Reducing dimensionality with PCA...")
    pca = PCA(n_components=2)
    return pca.fit_transform(features)


def plot_clusters(df: pd.DataFrame, pca_result: np.ndarray, labels: np.ndarray):
    """
    Visualizes the song clusters on a 2D scatter plot.

    Args:
        df: The original DataFrame with song info.
        pca_result: The 2D PCA-transformed data.
        labels: The cluster labels from K-Means.
    """
    print("Generating plot...")
    # Create a DataFrame for easy plotting.
    cluster_df = pd.DataFrame({
        'song': df['title'],
        'writers': df['writers'].apply(lambda x: ', '.join(x)),
        'pca1': pca_result[:, 0],
        'pca2': pca_result[:, 1],
        'cluster': labels
    })

    plt.figure(figsize=(14, 12))

    # Assign a unique color to each writer for the plot.
    unique_writers = sorted(
        list(set(writer for writers in df['writers'] for writer in writers)))
    writer_colors = {writer: color for writer, color in zip(
        unique_writers, sns.color_palette("hsv", len(unique_writers)))}

    # Plot each song as a point, colored by its primary writer.
    for i, row in cluster_df.iterrows():
        primary_writer = df['writers'].iloc[i][0]
        plt.scatter(row['pca1'], row['pca2'],
                    color=writer_colors[primary_writer], alpha=0.7, s=60)

    # Draw circles around the clusters to highlight them.
    for cluster_id in range(len(np.unique(labels))):
        clustered_points = cluster_df[cluster_df['cluster'] == cluster_id]
        center_x = clustered_points['pca1'].mean()
        center_y = clustered_points['pca2'].mean()
        # Calculate radius based on the mean standard deviation of the points.
        radius = clustered_points[['pca1', 'pca2']].std().mean()

        circle = plt.Circle((center_x, center_y), radius,
                            color='gray', fill=False, linestyle='--', linewidth=1.5)
        plt.gca().add_artist(circle)

    # Create a custom legend for writers.
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=writer,
                          markerfacecolor=writer_colors[writer], markersize=10) for writer in unique_writers]
    plt.legend(handles=handles, title="Writers",
               loc='upper right', fontsize='medium')

    # Configure plot aesthetics.
    plt.title('Song Clustering Based on Lyrical Style and Sentiment', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def main():
    """
    Main function to run the clustering analysis pipeline.
    """
    # Define the number of clusters to find.
    num_clusters = 3

    # Run the full pipeline.
    df = load_data(SINGLES_FILEPATH)
    df = preprocess_data(df)
    features = create_features(df)
    cluster_labels = perform_clustering(features, num_clusters)
    pca_result = reduce_dimensionality(features)
    plot_clusters(df, pca_result, cluster_labels)


if __name__ == "__main__":
    main()
