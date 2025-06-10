import pandas as pd
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from pathlib import Path

def from_str_to_list(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = df[column].apply(
        lambda s: np.fromstring(s.strip("[]"), sep=",").tolist()
    )
    return df

def min_profile_condition(row: pd.Series, profile_cols: list) -> bool:
    return all(min(row[col]) > -0.5 for col in profile_cols)

def process_ryos_dataset(input_path: Path, output_csv_path: Path, output_fasta_path: Path, version: int) -> None:
    # Définition des colonnes selon la version
    if version == 1:
        profile_cols = ['reactivity', 'deg_pH10', 'deg_50C', 'deg_Mg_pH10', 'deg_Mg_50C']
        s2n_cols = ['signal_to_noise_reactivity', 'signal_to_noise_deg_pH10',
                    'signal_to_noise_deg_50C', 'signal_to_noise_deg_Mg_pH10',
                    'signal_to_noise_deg_Mg_50C']
    else:
        profile_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']
        s2n_cols = ['signal_to_noise_reactivity', 'signal_to_noise_deg_Mg_pH10',
                    'signal_to_noise_deg_Mg_50C']

    # Chargement du CSV
    df = pd.read_csv(input_path, sep=';')
    original_cols = df.columns

    for col in profile_cols:
        df = from_str_to_list(df, col)

    mask_valid = df.apply(lambda row: min_profile_condition(row, profile_cols), axis=1)
    mask_s2n = df[s2n_cols].mean(axis=1) > 1.0
    df_filtered = df[mask_valid & mask_s2n].copy().reset_index(drop=True)

    # Sauvegarde CSV
    df_filtered[original_cols].to_csv(output_csv_path, sep=';', index=False)
    print(f"{len(df_filtered)} rows saved to: {output_csv_path}")

    # Sauvegarde FASTA
    records = [
        SeqRecord(Seq(seq), id=str(seq_id), description="")
        for seq, seq_id in zip(df_filtered["sequence"], df_filtered["ID"])
    ]
    SeqIO.write(records, output_fasta_path, "fasta")
    print(f"{len(records)} sequences written to: {output_fasta_path}")

if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[1]
    data_raw = root_dir / "data" / "raw"
    data_cleaned = root_dir / "data" / "cleaned"

    process_ryos_dataset(
        input_path=data_raw / "RYOS_I_full.csv",
        output_csv_path=data_cleaned / "cleaned_RYOS_I.csv",
        output_fasta_path=data_cleaned / "cleaned_RYOS_I.fasta",
        version=1
    )

    process_ryos_dataset(
        input_path=data_raw / "RYOS_II_full.csv",
        output_csv_path=data_cleaned / "cleaned_RYOS_II.csv",
        output_fasta_path=data_cleaned / "cleaned_RYOS_II.fasta",
        version=2
    )