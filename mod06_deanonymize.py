import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    # Merge the anonymized dataset with the auxiliary dataset on quasi-identifiers
    merged = pd.merge(anon_df, aux_df, on=["age", "zip3", "gender"], how="inner")

    # Count how many auxiliary records match each anon_id
    match_counts = merged.groupby("anon_id").size().reset_index(name="count")
    
    # Filter for unique matches (count == 1)
    unique_anon_ids = match_counts[match_counts["count"] == 1]["anon_id"]
    unique_matches = merged[merged["anon_id"].isin(unique_anon_ids)]
    
    # Select and rename columns as required
    result = unique_matches[["anon_id", "name"]].rename(columns={"name": "matched_name"})

    return result


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    if len(anon_df) == 0:
        return 0.0
    return len(matches_df) / len(anon_df)
