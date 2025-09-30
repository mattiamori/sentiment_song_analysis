# bert_classifier_v1.py

import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.utils import shuffle
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed
)
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# --- Global Constants ---
SINGLES_FILEPATH = "../srcs/single_writers_list.csv"
RESULTS_FILEPATH = "./results/results.csv"
SEPARATOR_STRING = "[SEP*]"

# --- Configuration ---
# If True, splits text by SEPARATOR_STRING for sentiment analysis.
USE_SEPARATOR_LOGIC = True
BATCH_SIZE = 30
EPOCHS = 10
EXECUTIONS = 5  # Number of times to run the entire training process.
MODEL_ID = "bert-base-cased"

# --- Feature Sets ---
# Define different combinations of features to be tested.
# The values will be combined into a single string for BERT input.
FEATURE_SETS = {
    'sentiment': ['text', 'pos', 'neg', 'neu', 'year', 'compound'],
    'only-pos-neg-neu': ['text', 'pos', 'neg', 'neu'],
    'only-year-compound': ['text', 'year', 'compound'],
    'only-text': ['text'],
    'only-year': ['text', 'year'],
    'only-compound': ['text', 'compound']
}


def get_or_create_writer_index(writer_name: str, writer_map: dict) -> int:
    """
    Assigns a unique integer index to each writer.

    Args:
        writer_name: The name of the writer.
        writer_map: A dictionary mapping writer names to indices.

    Returns:
        The index for the given writer.
    """
    if writer_name not in writer_map:
        writer_map[writer_name] = len(writer_map)
    return writer_map[writer_name]


def clean_text(text: str) -> str:
    """
    Removes special characters and the separator string from text.

    Args:
        text: The input text.

    Returns:
        The cleaned text.
    """
    chars_to_remove = set(",!@#?$%^&*()_+")
    text_cleaned = text.replace(SEPARATOR_STRING, "")
    text_cleaned = ''.join(c for c in text_cleaned if c not in chars_to_remove)
    return text_cleaned


def split_text_by_separator(text: str) -> list[str]:
    """
    Splits a long text into smaller chunks based on a custom separator.

    Args:
        text: The input text containing separators.

    Returns:
        A list of text chunks.
    """
    # This function is complex and specific to the dataset's structure.
    # It handles cases where the separator is a standalone word or attached to another word.
    text_array = []
    temp_text = ""
    chars_to_remove = set(",!@#?$%^&*()_+")
    for word in text.split():
        if word == SEPARATOR_STRING:
            # Clean and append the accumulated text chunk.
            temp_text = ''.join(
                c for c in temp_text if c not in chars_to_remove)
            if temp_text.strip():
                text_array.append(temp_text.strip())
            temp_text = ""
        elif SEPARATOR_STRING in word:
            # Handle cases like "word[SEP*]word"
            word_parts = word.split(SEPARATOR_STRING)
            temp_text += " " + word_parts[0]
            temp_text = ''.join(
                c for c in temp_text if c not in chars_to_remove)
            if temp_text.strip():
                text_array.append(temp_text.strip())

            temp_text = word_parts[1] if len(word_parts) > 1 else ""
        else:
            temp_text += " " + word

    # Add the last remaining text chunk.
    if temp_text.strip():
        temp_text = ''.join(c for c in temp_text if c not in chars_to_remove)
        text_array.append(temp_text.strip())

    return text_array


def compute_metrics(pred) -> dict:
    """
    Computes and returns a dictionary of metrics (accuracy, f1, precision, recall).
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Using 'macro' average is suitable for multi-class classification,
    # especially if there is a class imbalance.
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro')
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }


def train_and_evaluate(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    test_dataset: Dataset,
    num_labels: int,
    dt_string: str,
    suffix: str
):
    """
    Initializes, trains, and evaluates the BERT model.

    Args:
        train_dataset: The tokenized training dataset.
        eval_dataset: The tokenized evaluation dataset.
        test_dataset: The tokenized test dataset.
        num_labels: The number of unique authors (classes).
        dt_string: A timestamp string for identifying the training run.
        suffix: A string identifying the feature set used (e.g., 'sentiment').
    """
    # Load the pre-trained BERT model for sequence classification.
    model = BertForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=num_labels,
        # Useful if fine-tuning a model with a different head.
        ignore_mismatched_sizes=True
    )

    output_folder = f"./trainer/{dt_string}_{suffix}"

    # Define training arguments.
    training_args = TrainingArguments(
        output_dir=output_folder,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",  # Save model at the end of each epoch.
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        overwrite_output_dir=True,
        log_level="warning",
    )

    # Initialize the Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"--- Training model for feature set: {suffix} ---")
    trainer.train()

    # Make predictions on the test set.
    predictions = trainer.predict(test_dataset=test_dataset)
    predicted_labels = predictions.predictions.argmax(-1)
    true_labels = predictions.label_ids

    # --- Analysis and Reporting ---
    correct = np.sum(predicted_labels == true_labels)
    total = len(true_labels)
    accuracy = correct / total

    print(f"Test Set Accuracy: {accuracy:.2%}")

    # Create and save a pie chart of prediction results.
    save_img_path = os.path.join(output_folder, "prediction_summary.png")
    os.makedirs(os.path.dirname(save_img_path), exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.pie(
        [correct, total - correct],
        labels=['Correct', 'Incorrect'],
        colors=['#66b3ff', '#ff9999'],
        autopct='%1.1f%%',
        startangle=90
    )
    plt.title(f'Prediction Distribution: {suffix}')
    plt.savefig(save_img_path)
    plt.close()

    # Save results to a CSV file.
    save_results_to_csv({
        'train_id': dt_string,
        'subject': suffix,
        'accuracy': f"{accuracy:.4f}".replace(".", ","),
        'correct': correct,
        'incorrect': total - correct,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
    })


def save_results_to_csv(data: dict):
    """
    Appends a new row of results to the results CSV file.
    """
    directory = os.path.dirname(RESULTS_FILEPATH)
    if not os.path.exists(directory):
        os.makedirs(directory)

    df = pd.DataFrame([data])

    # If the file exists, append without the header. Otherwise, create it with a header.
    if os.path.exists(RESULTS_FILEPATH):
        df.to_csv(RESULTS_FILEPATH, mode='a', header=False, index=False)
    else:
        df.to_csv(RESULTS_FILEPATH, mode='w', header=True, index=False)


def run_experiment():
    """
    Main function to load data, preprocess it, and run the training
    and evaluation cycle for each defined feature set.
    """
    # Setup: download NLTK data and initialize sentiment analyzer.
    nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()

    # Set a seed for reproducibility.
    current_timestamp = int(datetime.now().timestamp())
    set_seed(current_timestamp)
    torch.manual_seed(current_timestamp)

    # Load and shuffle the dataset.
    df = pd.read_csv(SINGLES_FILEPATH).dropna(
        subset=['text', 'writers', 'year'])
    dataframe = shuffle(df, random_state=current_timestamp)

    # --- Data Preprocessing ---
    # This section prepares the data by splitting texts and calculating sentiment scores.
    all_texts, all_labels, all_sentiments, all_years = [], [], [], []
    writer_map = {}

    for _, row in dataframe.iterrows():
        writer = row["writers"]
        text = row["text"]
        year = row['year']

        writer_index = get_or_create_writer_index(writer, writer_map)

        if USE_SEPARATOR_LOGIC:
            chunks = split_text_by_separator(text)
            for chunk in chunks:
                sentiment_score = sid.polarity_scores(chunk)
                all_labels.append(writer_index)
                all_texts.append(chunk)
                all_sentiments.append(sentiment_score)
                all_years.append(int(year))
        else:
            cleaned_text = clean_text(text)
            sentiment_score = sid.polarity_scores(cleaned_text)
            all_labels.append(writer_index)
            all_texts.append(cleaned_text)
            all_sentiments.append(sentiment_score)
            all_years.append(int(year))

    # Prepare a dictionary with all processed data.
    data = {
        'text': all_texts,
        'pos': [s['pos'] for s in all_sentiments],
        'neu': [s['neu'] for s in all_sentiments],
        'neg': [s['neg'] for s in all_sentiments],
        'compound': [s['compound'] for s in all_sentiments],
        'year': all_years
    }

    num_labels = len(writer_map)
    dt_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    tokenizer = BertTokenizer.from_pretrained(MODEL_ID)

    # --- Training Loop for Each Feature Set ---
    for key, features in FEATURE_SETS.items():
        # Combine the selected features into a single string for each sample.
        # Example: "text:some lyrics | pos:0.1 | neg:0.2 ..."
        combined_texts = []
        for i in range(len(data['text'])):
            feature_strings = [f"{feat}:{data[feat][i]}" for feat in features]
            combined_texts.append(" | ".join(feature_strings))

        # Create a Hugging Face Dataset.
        dataset = Dataset.from_dict(
            {'text': combined_texts, 'labels': all_labels})

        # Tokenize the dataset.
        def tokenize_function(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Split into train, validation, and test sets.
        train_test_split = tokenized_dataset.train_test_split(
            test_size=0.2, seed=current_timestamp)
        test_eval_split = train_test_split['test'].train_test_split(
            test_size=0.5, seed=current_timestamp)

        train_dataset = train_test_split['train']
        eval_dataset = test_eval_split['test']
        test_dataset = test_eval_split['train']

        # Start the training process for this feature set.
        train_and_evaluate(train_dataset, eval_dataset,
                           test_dataset, num_labels, dt_string, key)


if __name__ == "__main__":
    for i in range(EXECUTIONS):
        print(f"\n--- STARTING EXECUTION {i + 1} of {EXECUTIONS} ---\n")
        run_experiment()
