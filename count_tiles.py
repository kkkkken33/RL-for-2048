import csv
from collections import Counter

def count_highest_values(file_path):
    """
    Reads a CSV file and counts occurrences of each value in the 'highest' column.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        dict: A dictionary with values from the 'highest' column as keys and their counts as values.
    """
    highest_counts = Counter()

    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if 'highest' not in reader.fieldnames:
                raise ValueError("The 'highest' column is not found in the CSV file.")

            for row in reader:
                highest_counts[row['highest']] += 1

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return highest_counts

if __name__ == "__main__":
    file_path = "episode_metrics.csv"  # Update this path if the file is located elsewhere
    counts = count_highest_values(file_path)

    if counts:
        print("Occurrences of each value in the 'highest' column:")
        for value, count in counts.items():
            print(f"{value}: {count}")