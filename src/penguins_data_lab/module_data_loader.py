import pandas as pd
import seaborn as sns

class DataLoaderCleaner:
    """Load and clean Palmer Penguins (via seaborn)."""

    def __init__(self):
        pass  # placeholder for future options

    def load_penguins(self) -> pd.DataFrame:
        """Load seaborn’s built-in penguins dataset."""
        df = sns.load_dataset("penguins")  # no external files needed
        return df

    def clean_penguins(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic cleaning: drop NAs, set categories, remove invalid values."""
        key_numeric = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

        # drop rows missing essential fields
        df_clean = df.dropna(subset=key_numeric + ["species", "island", "sex"]).copy()

        # cast strings to categories
        for col in ["species", "island", "sex"]:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype("category")

        # remove non-positive measurements
        for col in key_numeric:
            if col in df_clean.columns:
                df_clean = df_clean[df_clean[col] > 0]

        df_clean.reset_index(drop=True, inplace=True)  # tidy index after filtering
        return df_clean
