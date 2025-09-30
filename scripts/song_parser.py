# song_parser.py

import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Global File Paths ---
ORIGINAL_SONGS_FILEPATH = "../srcs/top_songs_original.csv"
CROPPED_SONGS_FILEPATH = "../srcs/list_cropped.csv"
SINGLE_WRITER_FILEPATH = "../srcs/single_writers_list.csv"
MULTI_WRITER_FILEPATH = "../srcs/multi_writers_list.csv"
RESULTS_FILEPATH = "./results/results.csv"

# Columns to drop from the original dataset.
USELESS_COLUMNS = ["description", "appears on",
                   "artist", "producer", "released", "streak"]


def load_csv(path: str, dtype=None) -> pd.DataFrame:
    """Safely loads a CSV file into a pandas DataFrame."""
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return pd.DataFrame()
    return pd.read_csv(path, dtype=dtype)


def create_cropped_list():
    """
    Reads the original CSV, drops unnecessary columns, and saves a cropped version.
    """
    print("Creating cropped list from original dataset...")
    df = load_csv(ORIGINAL_SONGS_FILEPATH)
    if df.empty:
        return

    df = df.drop(columns=USELESS_COLUMNS, errors='ignore')
    df.to_csv(CROPPED_SONGS_FILEPATH, index=False)
    print(f"Cropped dataset saved to '{CROPPED_SONGS_FILEPATH}'.")


def create_split_writer_csvs():
    """
    Splits the cropped dataset into two files: one for songs with a single writer
    and one for songs with multiple writers.
    """
    print("Splitting dataset by number of writers...")
    df = load_csv(CROPPED_SONGS_FILEPATH)
    if df.empty:
        return

    # Ensure 'writers' column is string type to prevent errors
    df['writers'] = df['writers'].astype(str)

    # Filter for single-writer songs
    df_single = df[df['writers'].apply(
        lambda x: len(x.split(',')) == 1)].copy()

    # Filter for multi-writer songs
    df_multi = df[df['writers'].apply(lambda x: len(x.split(',')) > 1)].copy()

    df_single.to_csv(SINGLE_WRITER_FILEPATH, index=False)
    df_multi.to_csv(MULTI_WRITER_FILEPATH, index=False)
    print(f"Single-writer songs saved to '{SINGLE_WRITER_FILEPATH}'.")
    print(f"Multi-writer songs saved to '{MULTI_WRITER_FILEPATH}'.")


def add_new_song():
    """
    Prompts the user for song details and adds a new song to the dataset.
    """
    print("\n--- Add a New Song ---")
    title = input('Title: ')
    writers = input('Writers (comma-separated if multiple): ')
    year = input("Year: ")
    song_text = input("Lyrics: ")

    df = load_csv(CROPPED_SONGS_FILEPATH)
    if df.empty:
        return

    # Check for duplicates
    if not df[(df['title'] == title) & (df['writers'] == writers)].empty:
        print("\nError: A song with the same title and writer(s) already exists. Aborting.")
        return

    new_item = {"title": title, "writers": writers,
                "year": str(year), "text": song_text}
    new_df = pd.concat([df, pd.DataFrame([new_item])], ignore_index=True)

    # Sort and save
    new_df = new_df.sort_values(by=['writers'], ascending=True)
    new_df.to_csv(CROPPED_SONGS_FILEPATH, index=False)
    print(f"\nSong '{title}' added successfully.")

    # Update split CSVs
    create_split_writer_csvs()


def modify_song():
    """
    Finds a song by title and allows the user to modify its details.
    """
    print("\n--- Modify a Song ---")
    search_title = input('Enter the title of the song to modify: ')
    df = load_csv(CROPPED_SONGS_FILEPATH)
    if df.empty:
        return

    matches = df[df['title'].str.contains(search_title, case=False, na=False)]
    if matches.empty:
        print(f"No song found with title containing '{search_title}'.")
        return

    for index, row in matches.iterrows():
        print(
            f"\nFound song: Title: {row['title']}, Writers: {row['writers']}")
        choice = input("Do you want to modify this song? (y/n): ").lower()
        if choice == 'y':
            print("(Leave blank to keep current value)")
            new_title = input(f"New title [{row['title']}]: ") or row['title']
            new_writers = input(
                f"New writers [{row['writers']}]: ") or row['writers']
            new_year = input(f"New year [{row['year']}]: ") or row['year']
            new_text = input(
                f"New text [{row['text'][:50]}...]: ") or row['text']

            df.loc[index] = [new_title, new_writers, row.get(
                'description'), new_text, new_year] + list(row[5:])

            df.to_csv(CROPPED_SONGS_FILEPATH, index=False)
            print("Song successfully modified.")
            create_split_writer_csvs()
            return  # Exit after modifying one song

    print("No more matching songs found or modification cancelled.")


def update_sentiment_scores():
    """
    Calculates and updates sentiment scores (pos, neg, neu, compound) for all songs.
    """
    print("Updating sentiment scores...")
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')

    sid = SentimentIntensityAnalyzer()
    df = load_csv(CROPPED_SONGS_FILEPATH)
    if df.empty:
        return

    df['text'] = df['text'].astype(str)

    # Clean text for sentiment analysis by removing the custom separator
    cleaned_texts = df['text'].str.replace("[SEP*]", "", regex=False)
    sentiment_scores = cleaned_texts.apply(
        lambda text: sid.polarity_scores(text))

    # Update DataFrame with scores
    df['pos'] = sentiment_scores.apply(lambda score: score['pos'])
    df['neg'] = sentiment_scores.apply(lambda score: score['neg'])
    df['neu'] = sentiment_scores.apply(lambda score: score['neu'])
    df['compound'] = sentiment_scores.apply(lambda score: score['compound'])

    df.to_csv(CROPPED_SONGS_FILEPATH, index=False)
    print("Sentiment scores updated successfully.")
    create_split_writer_csvs()


def plot_accuracy_distribution():
    """
    Loads training results and plots the accuracy distribution for each 'subject' (feature set),
    highlighting outliers.
    """
    print("Plotting accuracy distribution from training results...")
    df = load_csv(RESULTS_FILEPATH)
    if df.empty:
        return

    # Convert accuracy to float, handling comma decimal separator
    df['accuracy'] = df['accuracy'].str.replace(',', '.').astype(float)

    # Order subjects by median accuracy for better visualization
    sorted_subjects = df.groupby(
        'subject')['accuracy'].median().sort_values().index

    plt.figure(figsize=(15, 10))
    sns.set(style="whitegrid")

    # Boxplot to show distribution
    sns.boxplot(data=df, x='accuracy', y='subject',
                color='lightblue', order=sorted_subjects)

    # Stripplot to show individual data points
    sns.stripplot(data=df, x='accuracy', y='subject', color='black',
                  jitter=True, size=5, order=sorted_subjects)

    plt.title('Accuracy Distribution per Subject', fontsize=16)
    plt.xlabel('Accuracy', fontsize=12)
    plt.ylabel('Subject (Feature Set)', fontsize=12)
    plt.show()


def main_menu():
    """
    Displays the main menu and handles user input for script operations.
    """
    while True:
        print("\n--- Song Data Management and Analysis ---")
        menu_text = (
            "1.  Add New Song\n"
            "2.  Modify Song\n"
            "3.  Update Sentiment Scores\n"
            "4.  Recreate Cropped and Split CSVs\n"
            "5.  Plot Accuracy Distribution from Results\n"
            "6.  Exit\n"
        )
        print(menu_text)

        try:
            choice = int(input("Enter your choice: "))
            if choice == 1:
                add_new_song()
            elif choice == 2:
                modify_song()
            elif choice == 3:
                update_sentiment_scores()
            elif choice == 4:
                create_cropped_list()
                create_split_writer_csvs()
            elif choice == 5:
                plot_accuracy_distribution()
            elif choice == 6:
                print("Exiting.")
                break
            else:
                print("Invalid choice. Please enter a number from 1 to 6.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main_menu()
