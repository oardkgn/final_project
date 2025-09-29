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

    def plot_island_feature_distributions(
    self,
    df: pd.DataFrame,
    features=None,
    kind: str = "violin",
    add_points: bool = True,
):
        """Facet distributions of body features by island."""
        if "island" not in df.columns:
            raise KeyError("Expected 'island' column in DataFrame.")  # must have grouping col

        # default feature set
        if features is None:
            features = ["body_mass_g", "bill_length_mm", "bill_depth_mm", "flipper_length_mm"]

        # keep needed cols and drop NAs
        cols = ["island"] + [c for c in features if c in df.columns]
        data = df[cols].dropna().copy()

        if len(cols) <= 1:
            raise ValueError("No valid features found to plot.")  # nothing to facet

        # wide → long for faceting (one subplot per feature)
        long = data.melt(id_vars="island", value_vars=features, var_name="feature", value_name="value")

        # choose plot type
        if kind not in {"violin", "box", "boxen"}:
            raise ValueError("kind must be 'violin', 'box', or 'boxen'")

        # faceted categorical plot: x=island, y=value, col=feature
        g = sns.catplot(
            data=long,
            x="island",
            y="value",
            col="feature",
            kind=kind,
            sharey=False,      # separate scales per feature
            height=4,
            aspect=0.9,
            cut=0 if kind == "violin" else None,
            inner=None if kind == "violin" else None,
        )

        # optional raw points overlay for detail
        if add_points:
            for ax, feature in zip(g.axes.flat, features):
                sub = long[long["feature"] == feature]
                sns.stripplot(data=sub, x="island", y="value", dodge=True, alpha=0.45, size=3, ax=ax)

        # labels/titles
        g.set_axis_labels("Island", "Measurement value")
        g.set_titles("{col_name}")
        g.fig.subplots_adjust(top=0.85)
        g.fig.suptitle("Penguin Body Measurements by Island")
        return g


    def plot_kmeans_clusters(self, df_labeled: pd.DataFrame, x: str, y: str, label_col: str = "cluster"):
        """Scatter plot of two features colored by cluster labels."""
        fig, ax = plt.subplots(figsize=(6, 5))            # single axes
        sns.scatterplot(data=df_labeled, x=x, y=y, hue=label_col, ax=ax)  # color by cluster
        ax.set_title(f"K-Means Clusters by {x} vs {y}")
        fig.tight_layout()                                 # avoid clipping
        return ax


    