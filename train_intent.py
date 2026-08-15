# Retrains the intent classifier from data/portfolio_intents_clean.csv
# and overwrites models/intent_pipe.joblib. Run after editing the CSV
# to add/adjust training phrases for an intent.

import csv
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "portfolio_intents_clean.csv"
MODEL_PATH = BASE_DIR / "models" / "intent_pipe.joblib"

with CSV_PATH.open(encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

X = [r["query"].strip().lower() for r in rows]
y = [r["intent"].strip() for r in rows]

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=2000)),
    ("clf", LogisticRegression(C=5, max_iter=1000, random_state=42)),
])

scores = cross_val_score(pipe, X, y, cv=5)
print(f"5-fold CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

pipe.fit(X, y)
joblib.dump(pipe, MODEL_PATH)
print(f"Saved retrained model to {MODEL_PATH} ({len(X)} training examples)")
