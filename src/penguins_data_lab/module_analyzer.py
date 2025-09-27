from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

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
    
    def biggest_penguins_origin(self, df: pd.DataFrame, metric: str = "body_mass_g", group_by: str = "island", top_k: int = 10):
        """Among the top_k by `metric`, count occurrences of `group_by`. Returns (counts_dict, subset_df)."""
        if metric not in df.columns or group_by not in df.columns:
            raise KeyError("metric/group_by column not found.")
        # take top_k rows by metric (drop NA → sort desc → head)
        subset = df.dropna(subset=[metric, group_by]).sort_values(metric, ascending=False).head(top_k).copy()
        counts: Dict[str, int] = {}
        # count how many of those top_k belong to each group (e.g., island)
        for val in subset[group_by].astype(str):
            counts[val] = counts.get(val, 0) + 1
        return counts, subset

    def species_size_summary(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Return dict: species -> {mean_mass, mean_flipper} using loops."""
        required = ["species", "body_mass_g", "flipper_length_mm"]
        for col in required:
            if col not in df.columns:
                raise KeyError(f"Missing column: {col}")
        sums, counts = {}, {}
        # accumulate sums and counts per species
        for _, row in df.dropna(subset=required).iterrows():
            sp = str(row["species"])
            mass = float(row["body_mass_g"])
            fl = float(row["flipper_length_mm"])
            if sp not in sums:
                sums[sp] = {"mass": 0.0, "flipper": 0.0}
                counts[sp] = 0
            sums[sp]["mass"] += mass
            sums[sp]["flipper"] += fl
            counts[sp] += 1
        # convert accumulators to means
        result: Dict[str, Dict[str, float]] = {}
        for sp in sums:
            result[sp] = {
                "mean_mass": sums[sp]["mass"] / counts[sp],
                "mean_flipper": sums[sp]["flipper"] / counts[sp],
            }
        return result

    def add_size_class(self, df: pd.DataFrame, col: str = "body_mass_g") -> pd.DataFrame:
        """Add 'size_class' using if/elif/else thresholds (<3500 small, <=4500 medium, >4500 large)."""
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")
        out = df.copy()  # don’t mutate input
        classes: List[str] = []
        # assign class per row based on thresholds
        for val in out[col]:
            if pd.isna(val):
                classes.append("unknown")
            else:
                v = float(val)
                if v < 3500:
                    classes.append("small")
                elif v <= 4500:
                    classes.append("medium")
                else:
                    classes.append("large")
        out["size_class"] = classes
        return out

    def kmeans_on_numeric(self, df: pd.DataFrame, n_clusters: int = 3):
        """Fit K-Means on standardized numeric columns; returns (model, labeled_df)."""
        # select numeric features only
        num_df = df.select_dtypes(include=[np.number]).copy()
        if num_df.empty:
            raise ValueError("No numeric columns found for KMeans.")
        # simple z-score standardization
        num_df_std = (num_df - num_df.mean()) / num_df.std(ddof=0)
        # fit and predict clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init="auto")
        labels = kmeans.fit_predict(num_df_std)
        # attach labels to original df
        df_labeled = df.copy()
        df_labeled["cluster"] = labels.astype(int)
        return kmeans, df_labeled

