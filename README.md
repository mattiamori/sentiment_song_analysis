# Computational Analysis of Songwriter Stylometry using NLP and Deep Learning

This repository contains the source code for a Master's thesis project focused on computational stylometry and author attribution for song lyrics. The project leverages Natural Language Processing (NLP) techniques and deep learning models to identify and classify the unique lyrical styles of different songwriters.

## Table of Contents

- [Project Overview](#project-overview)
- [Core Objectives](#core-objectives)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)

## Project Overview

The primary goal of this research is to determine whether a songwriter can be accurately identified based solely on their lyrical patterns, sentiment, and structure. The project explores both unsupervised and supervised machine learning approaches to analyze and classify song lyrics.

- **Unsupervised Analysis:** K-Means clustering is used to discover natural groupings among songs based on their textual features (TF-IDF) and sentiment scores. This helps visualize inherent similarities and differences in lyrical styles without prior knowledge of the author.

- **Supervised Classification:** A pre-trained BERT (Bidirectional Encoder Representations from Transformers) model is fine-tuned for the task of author attribution. Different feature engineering strategies are tested to evaluate which combination of lyrical text, sentiment scores, and metadata (like the year of release) yields the best classification performance.

## Core Objectives

1.  **Data Preprocessing:** To build a robust pipeline for cleaning, structuring, and preparing song lyric data for machine learning tasks.
2.  **Sentiment Analysis:** To enrich the dataset by calculating sentiment scores (positive, negative, neutral, and compound) for each song or song segment using NLTK's VADER.
3.  **Unsupervised Clustering:** To apply K-Means clustering to identify stylistic patterns and group similar songs, providing insights into lyrical structures.
4.  **Supervised Classification:** To fine-tune a BERT model to classify a song's author, treating it as a multi-class text classification problem.
5.  **Feature Evaluation:** To experiment with different sets of input features for the BERT model to understand their impact on prediction accuracy.

## Methodology

### 1. Data Management and Preparation

The `song_parser.py` script serves as a command-line interface for managing the dataset. Its key functions include:

- Loading the raw song data from a CSV file.
- Cleaning the data by removing irrelevant columns.
- Calculating and appending sentiment scores for each lyric using NLTK's VADER sentiment analyzer.
- Splitting the main dataset into separate files for songs by single authors and those by multiple authors.
- Providing functionalities to add, modify, or inspect songs in the dataset.

### 2. Unsupervised Clustering (`cluster_analysis.py`)

This script performs a stylistic analysis using unsupervised learning:

- **Feature Extraction:** Song lyrics are converted into numerical vectors using `TfidfVectorizer`.
- **Feature Combination:** The TF-IDF vectors are combined with scaled numerical features (sentiment scores and year).
- **Clustering:** The K-Means algorithm is applied to the combined feature matrix to group songs into distinct clusters.
- **Visualization:** Principal Component Analysis (PCA) is used to reduce the dimensionality of the features to 2D, allowing the clusters to be visualized on a scatter plot. Each point is colored by its primary author to show the relationship between authorial style and the discovered clusters.

### 3. Supervised Classification (BERT Models)

Two primary experimental setups are used to fine-tune a `bert-base-cased` model for author attribution.

#### a) `bert_classifier_v1.py`: Combined Feature String Approach

In this version, all features (text, sentiment scores, year) are concatenated into a single, structured string.

- **Example Input:** `"text:some lyrics here | pos:0.12 | neg:0.05 | neu:0.83 | year:1998 | compound:0.45"`
- **Rationale:** This approach allows the BERT model to learn relationships between the text and its associated metadata within the same input sequence, leveraging its ability to understand context.

#### b) `bert_classifier_v2.py`: Text-Centric Approach

This script follows a more traditional approach to fine-tuning BERT for text classification.

- **Input:** The model is primarily trained on the raw lyrical text.
- **Method:** While the dataset contains other features (sentiment, year), a standard `BertForSequenceClassification` model will focus on the tokenized text inputs (`input_ids`, `attention_mask`). This setup serves as a baseline and tests the power of textual information alone. The additional features are kept in the dataset, allowing for future extensions with multi-modal model architectures.

## Repository Structure

```
.
├── srcs/                    # Directory for dataset files
│   ├── top_songs_original.csv   # Raw dataset (not included)
│   ├── list_cropped.csv         # Processed dataset
│   └── single_writers_list.csv  # Dataset with only single-author songs
│
├── results/                 # Output directory for training results
│   └── results.csv              # CSV log of model accuracy and metrics
│
├── trainer/                 # Output directory for saved models and logs
│
├── cluster_analysis.py      # Script for unsupervised K-Means clustering and visualization.
├── bert_classifier_v1.py    # BERT training script using the combined feature string method.
├── bert_classifier_v2.py    # BERT training script using the text-centric method.
├── song_parser.py           # CLI tool for dataset management and analysis.
└── README.md                # This file.
```

## Setup and Installation

**Prerequisites:** Python 3.8+ and pip.

1.  **Clone the repository:**

    ```bash
    git clone https://your-repository-url.git
    cd your-repository-directory
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    Create a `requirements.txt` file with the following content:
    ```
    pandas
    numpy
    scikit-learn
    matplotlib
    seaborn
    torch
    transformers
    datasets
    nltk
    ```
    Then, install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Management (`song_parser.py`)

Run this script to access an interactive menu for managing your dataset.

````bash
python song_parser.py
```You can use this to add new songs, modify existing ones, or update sentiment scores for the entire dataset.

### 2. Clustering Analysis (`cluster_analysis.py`)
To run the unsupervised clustering and generate the PCA plot:
```bash
python cluster_analysis.py
````

### 3. BERT Model Training

To run the author classification experiments, execute either of the BERT scripts. Each script will loop through the defined `FEATURE_SETS` and train a separate model for each.

**To run the combined feature string experiment:**

```bash
python bert_classifier_v1.py
```

**To run the text-centric experiment:**

```bash
python bert_classifier_v2.py
```

## Dataset

The project expects an initial CSV file located at `srcs/top_songs_original.csv`. This file should contain at least the following columns: `title`, `writers`, `year`, and `text`.

The `song_parser.py` script will generate the processed files (`list_cropped.csv`, `single_writers_list.csv`) required by the analysis and training scripts.

## Results

The performance metrics of each training run (accuracy, correct/incorrect predictions, etc.) are automatically logged in the `results/results.csv` file.

Each training session also creates a dedicated folder inside `trainer/` (for `v1`) or `trainer_v2/` (for `v2`), which contains:

- Model checkpoints.
- TensorBoard logs.
- A pie chart (`prediction_summary.png`) visualizing the test set accuracy.
