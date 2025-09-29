# penguins_data_lab

A small but complete, installable Python package that walks through a full mini data-science workflow on the **Palmer Penguins** dataset (via `seaborn`). The project is designed to be clear, educational, and easy to grade: it separates logic (analysis) from plotting (visualization), includes docstrings and comments, and ships with both a Markdown tutorial and a Jupyter notebook.

---

## What this project does (overview)

- **Loads & cleans** penguin measurements (bill length/depth, flipper length, body mass, species, island, sex).
- **Analyzes** the cleaned data with a mix of **core Python** (for/while, if/elif/else, dictionaries, tuples) and **pandas/numpy**.
- **Visualizes** distributions, correlations, clusters, and island-based differences.
- **Demonstrates ML** with a tiny **K-Means** example.
- **Packages** everything as a proper Python package with `pyproject.toml` and a `src/` layout, ready to install locally (`pip install -e .`).

---

## What’s inside (modules & classes)

### `DataLoaderCleaner` (in `module1.py`)
- `load_penguins()` — loads the dataset from `seaborn` (portable, no external files).
- `clean_penguins(df)` — drops essential NAs, casts `species/island/sex` to `category`, removes non-positive measurements, resets index.

### `Analyzer` (in `analyzer.py`)
- **Core Python features** on purpose:
  - `manual_value_counts(df, column)` — counts with a dict + `for` loop.
  - `find_extremes(df, value_col)` — manual `(min, max)` via a loop (returns a tuple).
  - `top_heaviest(df, top_n)` — builds a result with a `while` loop.
  - `add_size_class(df, col)` — uses `if/elif/else` thresholds to label `small/medium/large`.
- **Pandas/numpy & simple ML**:
  - `summary_stats(df)` — descriptive stats for numeric columns.
  - `species_size_summary(df)` — dict per species: mean mass & mean flipper length.
  - `biggest_penguins_origin(df, metric, group_by, top_k)` — among the heaviest `top_k`, where do they come from?
  - `kmeans_on_numeric(df, n_clusters)` — z-scores numeric columns, runs **K-Means** (scikit-learn), returns `(model, labeled_df)`.

### `Visualizer` (in `visualizer.py`)
- `plot_distributions(df)` — histograms for key numeric features.
- `plot_correlation_heatmap(df)` — annotated correlation matrix.
- `plot_kmeans_clusters(df_labeled, x, y, label_col)` — 2-D scatter colored by cluster.
- `plot_island_feature_distributions(df, features=None, kind="violin", add_points=True)` — facets per feature with islands on the x-axis to compare distributions.

---

## Python packages used

- **Core**: `python` (loops, conditionals, dicts, tuples, list comprehensions)
- **Data**: `pandas`, `numpy`
- **Plots**: `matplotlib`, `seaborn`
- **ML**: `scikit-learn` (K-Means)
- **Packaging**: `hatchling` via `pyproject.toml` (PEP 621)

---

## Basic operations showcased

- Data wrangling: NA handling, dtype casting, sanity filtering
- Summary statistics, manual counters, min/max scans
- Groupwise summaries, dictionary accumulators
- Visualization: distributions, correlations, clusters, faceted comparisons
- Simple unsupervised learning (K-Means) with standardization

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .
