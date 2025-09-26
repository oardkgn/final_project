from typing import Dict, Tuple
import pandas as pd
import numpy as np

class Analyzer:
    """Analysis utilities for Palmer Penguins (with core Python features)."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state  # for reproducibility

    def summary_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Descriptive stats for numeric columns."""
        # select numerics and summarize
        return df.select_dtypes(include=[np.number]).describe().T

    def manual_value_counts(self, df: pd.DataFrame, column: str) -> Dict[str, int]:
        """Counts via plain Python (no pandas .value_counts)."""
        counts: Dict[str, int] = {}
        for val in df[column].astype(str):  # normalize keys as str
            counts[val] = counts.get(val, 0) + 1
        return counts
    
    def find_extremes(self, df: pd.DataFrame, value_col: str) -> Tuple[float, float]:
        """Min/max using a manual loop; returns (min, max)."""
        series = df[value_col].dropna().tolist()
        if not series:
            raise ValueError(f"Column {value_col} has no numeric data.")
        min_val = max_val = float(series[0])  # init bounds
        for x in series[1:]:  # iterate over remaining values and update min/max bounds
            x = float(x)
            if x < min_val:
                min_val = x
            elif x > max_val:
                max_val = x
        return (min_val, max_val)

    def top_heaviest(self, df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """Top-N rows by body_mass_g using a while loop."""
        if "body_mass_g" not in df.columns:
            raise KeyError("Expected 'body_mass_g' in DataFrame.")
        # sort once, then pick first N via while
        df_sorted = df.sort_values("body_mass_g", ascending=False).reset_index(drop=True)
        rows = []
        i = 0
        while i < min(top_n, len(df_sorted)):
            rows.append(df_sorted.loc[i])
            i += 1
        return pd.DataFrame(rows)
