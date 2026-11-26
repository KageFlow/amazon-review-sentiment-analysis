# Amazon Product Review Sentiment Analysis

A beginner-friendly machine-learning project that classifies product reviews as **positive** or **negative** using Python and Scikit-learn.

## Objectives
- Load and clean review text with Pandas.
- Convert text into numerical features using TF-IDF.
- Split data into training and testing sets.
- Train a Logistic Regression classifier.
- Predict sentiment for new reviews.
- Evaluate with accuracy, F1-score, a classification report, and a confusion matrix.

## Repository structure
```
amazon-review-sentiment-analysis/
├── data/
│   ├── README.md
│   └── amazon_reviews_demo.csv
├── models/                 # Created when notebook runs
├── sentiment_analysis.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Run in Google Colab
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `sentiment_analysis.ipynb` or open it from GitHub.
3. Run cells from top to bottom.
4. Upload a CSV when prompted, or use the included demo dataset.

## Run locally
```bash
pip install -r requirements.txt
jupyter notebook sentiment_analysis.ipynb
```

## Dataset format
Your CSV must contain:
- `review`: review text
- `sentiment`: `positive` or `negative`

The included `data/amazon_reviews_demo.csv` is a small **synthetic demo dataset**, not actual Amazon customer reviews. For a meaningful project evaluation, replace it with a real, appropriately licensed dataset and document its source, license, and any preprocessing. Do not describe the demo data as real Amazon data.

## Method
1. Clean and normalize review text.
2. Split data with stratification.
3. Fit TF-IDF on training text only.
4. Train Logistic Regression.
5. Evaluate on the held-out test set.
6. Predict sentiment for new reviews.

## Metrics
- **Accuracy:** proportion of correct predictions.
- **F1-score:** harmonic mean of precision and recall for the positive class.
- **Confusion matrix:** counts correct and incorrect predictions by class.

A small demo dataset is only for checking that the notebook runs; its scores should not be treated as evidence of real-world model quality.

## Technologies
Python, Pandas, NumPy, Matplotlib, Scikit-learn, Google Colab.
