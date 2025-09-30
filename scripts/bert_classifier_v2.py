# bert_classifier_v2.py

import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset, DatasetDict
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
RESULTS_FILEPATH = "./results/results_v2.csv"
SEPARATOR_STRING = "[SEP*]"

# --- Configuration ---
USE_SEPARATOR_LOGIC = True
BATCH_SIZE = 15
EPOCHS = 10
EXECUTIONS = 5
MODEL_ID = "bert-base-cased"

# --- Feature Sets ---
# NOTE: For a standard BERT model, only the 'text' and 'label' columns are used
# by the Trainer. The other columns are kept for potential use with custom models
# but will be ignored by the default BertForSequenceClassification training process.
FEATURE_SETS = {
    'sentiment': ['text', 'label', 'pos', 'neg', 'neu', 'year', 'compound'],
    'only-pos-neg-neu': ['text', 'label', 'pos', 'neg', 'neu'],
    'only-year-compound': ['text', 'label', 'year', 'compound'],
    'only-text': ['text', 'label'],
    'only-year': ['text', 'label', 'year'],
    'only-compound': ['text', 'label', 'compound']
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


def split_text_by_separator(text: str) -> list[str]:
    """
    Splits a long text into smaller chunks based on a custom separator.
    """
    text_array = []
    temp_text = ""
    for word in text.split():
        if word == SEPARATOR_STRING:
            if temp_text.strip():
                text_array.append(temp_text.strip())
            temp_text = ""
        elif SEPARATOR_STRING in word:
            word_parts = word.split(SEPARATOR_STRING)
            temp_text += " " + word_parts[0]
            if temp_text.strip():
                text_array.append(temp_text.strip())
            temp_text = word_parts[1] if len(word_parts) > 1 else ""
        else:
            temp_text += " " + word

    if temp_text.strip():
        text_array.append(temp_text.strip())

    return text_array


def compute_metrics(pred) -> dict:
    """
    Computes and returns a dictionary of metrics.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)

    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


def train_and_evaluate(
    dataset_dict: DatasetDict,
    id2label: dict,
    label2id: dict,
    dt_string: str,
    suffix: str
):
    """
    Initializes, trains, and evaluates the BERT model.
    """
    num_labels = len(id2label)

    # Load model with label mappings.
    model = BertForSequenceClassification.from_pretrained(
        MODEL_ID,
        id2label=id2label,
        label2id=label2id,
        num_labels=num_labels
    )

    output_folder = f"./trainer_v2/{dt_string}_{suffix}"

    training_args = TrainingArguments(
        output_dir=output_folder,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        overwrite_output_dir=True,
        log_level="warning",
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['eval'],
        compute_metrics=compute_metrics,
    )

    print(f"--- Training model for feature set: {suffix} ---")
    trainer.train()

    # Predictions on the test set.
    predictions = trainer.predict(test_dataset=dataset_dict['test'])
    predicted_labels = predictions.predictions.argmax(-1)
    true_labels = predictions.label_ids

    # Analysis and Reporting.
    correct = np.sum(predicted_labels == true_labels)
    total = len(true_labels)
    accuracy = correct / total

    print(f"Test Set Accuracy: {accuracy:.2%}")

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

    if os.path.exists(RESULTS_FILEPATH):
        df.to_csv(RESULTS_FILEPATH, mode='a', header=False, index=False)
    else:
        df.to_csv(RESULTS_FILEPATH, mode='w', header=True, index=False)


def run_experiment():
    """
    Main function to load data, preprocess it, and run the training cycle.
    """
    nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()

    current_timestamp = int(datetime.now().timestamp())
    set_seed(current_timestamp)
    torch.manual_seed(current_timestamp)

    df = pd.read_csv(SINGLES_FILEPATH).dropna(
        subset=['text', 'writers', 'year'])
    dataframe = shuffle(df, random_state=current_timestamp)

    # --- Data Preprocessing ---
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
            # Fallback if separator logic is disabled
            cleaned_text = text.replace(SEPARATOR_STRING, "")
            sentiment_score = sid.polarity_scores(cleaned_text)
            all_labels.append(writer_index)
            all_texts.append(cleaned_text)
            all_sentiments.append(sentiment_score)
            all_years.append(int(year))

    data = {
        'text': all_texts,
        'pos': [s['pos'] for s in all_sentiments],
        'neu': [s['neu'] for s in all_sentiments],
        'neg': [s['neg'] for s in all_sentiments],
        'compound': [s['compound'] for s in all_sentiments],
        'year': all_years,
        'label': all_labels
    }

    # Create label mappings
    id2label = {i: name for name, i in writer_map.items()}
    label2id = writer_map

    dt_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    tokenizer = BertTokenizer.from_pretrained(MODEL_ID)

    def preprocess_function(examples):
        # Tokenize the text. The label is passed through untouched.
        tokenized_output = tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=128)
        tokenized_output["label"] = examples["label"]
        return tokenized_output

    # --- Training Loop for Each Feature Set ---
    for key, features in FEATURE_SETS.items():
        # Create a Hugging Face Dataset from the full data dictionary.
        full_dataset = Dataset.from_dict(data)

        # Remove columns that are not part of the current feature set.
        # This is done for experimental consistency, but BERT will only use the text.
        columns_to_remove = [
            col for col in full_dataset.column_names if col not in features]
        dataset = full_dataset.remove_columns(columns_to_remove)

        # Split into train, validation, and test sets.
        train_test_split = dataset.train_test_split(
            test_size=0.2, seed=current_timestamp)
        test_eval_split = train_test_split['test'].train_test_split(
            test_size=0.5, seed=current_timestamp)

        ds = DatasetDict({
            'train': train_test_split['train'],
            'test': test_eval_split['train'],
            'eval': test_eval_split['test']
        })

        # Apply tokenization to all splits.
        tokenized_ds = ds.map(preprocess_function, batched=True)

        train_and_evaluate(tokenized_ds, id2label, label2id, dt_string, key)


if __name__ == "__main__":
    for i in range(EXECUTIONS):
        print(f"\n--- STARTING EXECUTION {i + 1} of {EXECUTIONS} ---\n")
        run_experiment()
