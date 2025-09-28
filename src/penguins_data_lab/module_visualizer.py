import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    """Visualization utilities for the penguins dataset."""

    def plot_distributions(self, df: pd.DataFrame):
        """Plot histograms for key numeric columns."""
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 grid
        cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
        axes = axes.ravel()  # flatten to 1D for easy looping
        for ax, col in zip(axes, cols):
            if col in df.columns:  # plot only if column exists
                sns.histplot(df[col], bins=20, ax=ax, kde=True)
                ax.set_title(f"Distribution of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
            else:
                ax.set_visible(False)  # hide unused axes
        fig.tight_layout()  # fix spacing
        return axes[-1]

    def plot_correlation_heatmap(self, df: pd.DataFrame):
        """Plot a correlation heatmap for numeric columns."""
        corr = df.select_dtypes(include=[np.number]).corr()  # numeric-only correlation
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, fmt=".2f", ax=ax)  # show values on cells
        ax.set_title("Correlation Heatmap (Numeric Columns)")
        fig.tight_layout()
        return ax

    