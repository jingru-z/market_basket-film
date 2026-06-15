# Project 2 - Market-Basket Analysis on IMDb Movie Actors

This project uses the Kaggle dataset:

`harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows`

Each movie is treated as one basket. The items in the basket are the actors
listed in the columns `Star1`, `Star2`, `Star3`, and `Star4`.

The main technique is the Apriori algorithm, used to find frequent actor
itemsets and association rules.

## Notebook

The main notebook is:

- `market_basket_imdb.ipynb`

It is designed to run on Google Colab or Jupyter. Before running it, replace
the Kaggle placeholders with your own Kaggle API credentials:

```python
KAGGLE_USERNAME = "xxxxxx"
KAGGLE_KEY = "xxxxxx"
```

Before publishing the project, keep these values as `"xxxxxx"` and do not
commit real Kaggle credentials.

## Setup in VS Code

1. Open this folder in VS Code.
2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure Kaggle credentials.

Recommended option:

```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_key"
```

On Windows PowerShell:

```powershell
$env:KAGGLE_USERNAME="your_username"
$env:KAGGLE_KEY="your_key"
```

Alternatively, edit the placeholders in `market_basket_imdb.py` while running
locally. Before publishing the project, put the values back to `"xxxxxx"`.

5. Run the final experiment:

```bash
python3 market_basket_imdb.py
```

## Useful options

Run on a subset:

```bash
python3 market_basket_imdb.py --use-subsample --subsample-size 200
```

Change support and confidence:

```bash
python3 market_basket_imdb.py --min-support 0.002 --min-confidence 0.2
```

Run the threshold comparison experiment used in the report:

```bash
python3 market_basket_imdb.py --skip-download --compare-thresholds
```

If the dataset was already downloaded or manually placed under `data/raw`,
skip the Kaggle download step:

```bash
python3 market_basket_imdb.py --skip-download
```

## Outputs

The script writes the following files:

- `outputs/frequent_itemsets.csv`
- `outputs/association_rules.csv`
- `outputs/threshold_comparison.csv`

The final setting used in the report is:

- minimum support: `0.003`
- minimum confidence: `0.50`
- minimum support count: `3`

The threshold comparison includes:

| Setting | Min support | Min confidence | Support count | Itemsets | Rules |
|---|---:|---:|---:|---:|---:|
| High support | 0.020 | 0.50 | 20 | 0 | 0 |
| Loose | 0.002 | 0.20 | 2 | 791 | 439 |
| Final | 0.003 | 0.50 | 3 | 299 | 46 |

## Method

The implemented technique is Apriori. It works level by level:

1. Count frequent single actors.
2. Generate candidate actor pairs only from frequent actors.
3. Keep only candidates whose support is above the threshold.
4. Repeat for itemsets of size 3 and 4.
5. Derive association rules from the frequent itemsets.

The implementation stores transactions as Python sets and counts only
combinations that appear inside each basket. Since this specific dataset has
four actor columns, each transaction has at most four items, which keeps
candidate counting compact while still using the standard Apriori principle.
