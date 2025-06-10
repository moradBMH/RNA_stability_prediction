import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from pathlib import Path


def load_pairwise_data(pairwise_path):
    df = pd.read_csv(pairwise_path, sep="\t", header=None)
    df.columns = [
        "seq1", "seq2", "identity", "align_length", "mismatches", "gap_opens",
        "q_start", "q_end", "s_start", "s_end", "evalue", "bit_score"
    ]
    df["seq1"] = df["seq1"].astype(str)
    df["seq2"] = df["seq2"].astype(str)
    return df


def compute_distance_matrix(df):
    unique_ids = pd.unique(pd.concat([df["seq1"], df["seq2"]]))
    id_to_index = {seq_id: idx for idx, seq_id in enumerate(unique_ids)}
    n = len(unique_ids)

    dist_matrix = np.ones((n, n))
    np.fill_diagonal(dist_matrix, 0.0)

    for _, row in df.iterrows():
        i = id_to_index[row["seq1"]]
        j = id_to_index[row["seq2"]]
        dist = 1.0 - float(row["identity"]) / 100.0
        dist_matrix[i, j] = dist
        dist_matrix[j, i] = dist

    return dist_matrix, unique_ids


def evaluate_test_split(dist_matrix, unique_ids, n_test=440):
    n = dist_matrix.shape[0]
    d_min = []

    for i in range(n):
        row = np.copy(dist_matrix[i])
        row[i] = np.inf
        d_min.append(np.min(row))

    d_min = np.array(d_min)
    sorted_indices = np.argsort(d_min)[::-1]
    test_indices = sorted_indices[:n_test]
    rest_indices = sorted_indices[n_test:]

    dists_test = d_min[test_indices]
    dists_rest = d_min[rest_indices]

    # Statistiques
    overlap = np.sum(dists_test <= np.max(dists_rest))
    overlap_rate = overlap / n_test
    u_stat, p_value = mannwhitneyu(dists_test, dists_rest, alternative='greater')

    print("\nMann–Whitney U test (H1: test set more distant than rest)")
    print(f"U statistic = {u_stat}")
    print(f"p-value     = {p_value:.5f}")
    print(f"Mean d_min (test) = {np.mean(dists_test):.4f}")
    print(f"Mean d_min (rest) = {np.mean(dists_rest):.4f}")
    print(f"Overlap    = {overlap} / {n_test} ({overlap_rate:.2%})")

    plt.figure(figsize=(8, 5))
    plt.hist(dists_test, bins=30, alpha=0.7, label="Test set", color="tomato", edgecolor="black")
    plt.hist(dists_rest, bins=30, alpha=0.5, label="Rest of dataset", color="skyblue", edgecolor="black")
    plt.axvline(np.mean(dists_test), color="red", linestyle="--", label="Test mean")
    plt.axvline(np.mean(dists_rest), color="blue", linestyle="--", label="Rest mean")
    plt.title("Minimum distance to any other sequence")
    plt.xlabel("Min pairwise distance")
    plt.ylabel("Number of sequences")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return [unique_ids[i] for i in test_indices]


def export_test_and_train_sequences(feature_path, test_output_path, train_output_path, test_ids):
    df = pd.read_csv(feature_path, sep=';')
    df_test = df[df['ID'].astype(str).isin(set(test_ids))]
    df_train = df[~df['ID'].astype(str).isin(set(test_ids))]

    df_test.to_csv(test_output_path, sep=';', index=False)
    df_train.to_csv(train_output_path, sep=';', index=False)

    print(f"{len(df_test)} test sequences saved to: {test_output_path}")
    print(f"{len(df_train)} train sequences saved to: {train_output_path}")


def run_analysis(pairwise_path, feature_path, test_output_path, train_output_path, n_test=440):
    df = load_pairwise_data(pairwise_path)
    dist_matrix, unique_ids = compute_distance_matrix(df)
    test_ids = evaluate_test_split(dist_matrix, unique_ids, n_test=n_test)
    export_test_and_train_sequences(feature_path, test_output_path, train_output_path, test_ids)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]

    # === RYOS I (seul utilisé pour split train/test)
    run_analysis(
        pairwise_path=base_dir / "data/similarity_matrices/pairwise_RYOS_I.tsv",
        feature_path=base_dir / "data/cleaned/cleaned_RYOS_I.csv",
        test_output_path=base_dir / "data/processed/val_set_RYOS_I.csv",
        train_output_path=base_dir / "data/processed/train_RYOS_I.csv"
    )