import os
import json
import pandas as pd
from datasets import Dataset, DatasetDict
from pathlib import Path
import re

# Paths to data
train_csv_path = "train_new.csv"
valid_csv_path = "val_new.csv"
test_csv_path = "test_new.csv"

# Output paths
output_dir = Path("./new_hf_datasets")
vocab_path = output_dir / "vocab.json"


# Function to read and process phoneme sequences
def load_word_sequences(csv_file):
    data = pd.read_csv(csv_file)
    entries = []
    

    for _, row in data.iterrows():
        wav_path = row["wav"]
        word_seq = row["expt_trans"].strip().lower()

        # Create an entry for the dataset
        entries.append({"audio": wav_path, "words": word_seq})

    return entries

# Create Hugging Face dataset from CSV
def create_hf_dataset(train_csv, valid_csv, test_csv):
    train_data = load_word_sequences(train_csv)
    valid_data = load_word_sequences(valid_csv)
    test_data = load_word_sequences(test_csv)

    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    valid_dataset = Dataset.from_pandas(pd.DataFrame(valid_data))
    test_dataset = Dataset.from_pandas(pd.DataFrame(test_data))

    return DatasetDict({"train": train_dataset, "valid": valid_dataset, "test": test_dataset})

# Function to build vocabulary from word sequences
def build_vocab(dataset):
    word_set = set()
    for split in ["train", "valid", "test"]:
        for example in dataset[split]:
            word_set.update(re.findall(r"[가-힣 ]", example["words"]))  # 전체 문장에서 음절만 추출

    # Add special tokens
    vocab = {word: idx for idx, word in enumerate(sorted(word_set))}
    vocab["<pad>"] = len(vocab)
    vocab["<unk>"] = len(vocab)

    return vocab

# Main script
if __name__ == "__main__":
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create Hugging Face datasets
    hf_dataset = create_hf_dataset(train_csv_path, valid_csv_path, test_csv_path)

    # Save datasets
    hf_dataset.save_to_disk(str(output_dir))

    # Build and save vocabulary
    vocab = build_vocab(hf_dataset)
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

    print(f"Datasets and vocabulary saved to {output_dir}")
