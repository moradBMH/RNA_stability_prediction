import pandas as pd
import json
import numpy as np
from pathlib import Path

def to_list(cell):
    """Convert a string-formatted list to a list of floats."""
    return [float(x) for x in cell.strip("[]").split(",")]

def pad_to_length(lst, target_length):
    """Pad list with None to match a specific length."""
    return lst + [None] * (target_length - len(lst))

def make_mask(length, valid_len):
    """Create a binary mask of given length with valid_len 1s and the rest 0s."""
    return [1] * valid_len + [0] * (length - valid_len)

def process_csv(filepath, target_cols, missing_cols):
    df = pd.read_csv(filepath, sep=';')
    entries = []

    for _, row in df.iterrows():
        seq = row['sequence'].replace("U", "T")
        structure = row['structure']
        ryos = row['RYOS']
        valid_len = len(to_list(row[target_cols[0]]))
        length = len(seq)

        entry = {
            "sequence": seq,
            "structure": structure,
            "mask": make_mask(length, valid_len),
            "RYOS": ryos
        }

        for col in target_cols:
            if col in df.columns:
                entry[col] = pad_to_length(to_list(row[col]), length)

        for col in missing_cols:
            entry[col] = [None] * length

        entries.append(entry)

    return entries

def write_json(entries, output_path):
    with open(output_path, "w") as f:
        json.dump(entries, f)
    print(f"✅ JSON saved to {output_path} ({len(entries)} entries)")

if __name__ == "__main__":
    # === Define root ===
    root_dir = Path(__file__).resolve().parents[2]
    processed_dir = root_dir / "data" / "processed"

    # === Common parameters for RYOS I ===
    target_cols = ['reactivity', 'deg_pH10', 'deg_50C', 'deg_Mg_pH10', 'deg_Mg_50C']
    missing_cols = []  # RYOS I has all targets

    # === Process TRAIN set ===
    train_path_csv = processed_dir / "train_RYOS_I.csv"
    train_path_json = processed_dir / "train_RYOS_I.json"
    train_data = process_csv(train_path_csv, target_cols, missing_cols)
    write_json(train_data, train_path_json)

    # === Process VALIDATION set ===
    val_path_csv = processed_dir / "val_set_RYOS_I.csv"
    val_path_json = processed_dir / "val_set_RYOS_I.json"
    val_data = process_csv(val_path_csv, target_cols, missing_cols)
    write_json(val_data, val_path_json)
