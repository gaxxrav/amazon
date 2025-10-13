import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

# Paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SR_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(SR_ROOT, 'dataset')
MODELS_DIR = os.path.join(SR_ROOT, 'models')
OUT_CSV = os.path.join(DATA_DIR, 'test_out.csv')

os.makedirs(MODELS_DIR, exist_ok=True)


def smape(y_true, y_pred) -> float:
	import numpy as np
	y_true = np.asarray(y_true, dtype=float)
	y_pred = np.asarray(y_pred, dtype=float)
	num = np.abs(y_true - y_pred)
	denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
	mask = denom == 0
	denom[mask] = 1.0
	return float(np.mean(num / denom))


def ensure_positive_price(value: float, minimum: float = 0.01) -> float:
	return float(max(value, minimum))


def _extract_text(df: pd.DataFrame) -> pd.Series:
	return df['catalog_content'].fillna('')


def _log_price(y: np.ndarray) -> np.ndarray:
	return np.log(np.clip(y, 0.01, None))


def _inv_log_price(y_log: np.ndarray) -> np.ndarray:
	return np.exp(y_log)


def train_and_eval(random_state: int = 42) -> Tuple[Pipeline, float]:
	train_csv = os.path.join(DATA_DIR, 'train.csv')
	df = pd.read_csv(train_csv)
	df = df.dropna(subset=['catalog_content', 'price'])

	y = df['price'].astype(float).values
	y_log = _log_price(y)

	X_train, X_val, y_train, y_val = train_test_split(df, y_log, test_size=0.2, random_state=random_state)

	text_transformer = Pipeline([
		('selector', FunctionTransformer(_extract_text, validate=False)),
		('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2))
	])

	preprocessor = ColumnTransformer([
		('text', text_transformer, ['catalog_content'])
	], remainder='drop')

	model = Ridge(alpha=1.0, random_state=random_state)

	pipe = Pipeline([
		('prep', preprocessor),
		('model', model)
	])

	pipe.fit(X_train, y_train)

	val_pred_log = pipe.predict(X_val)
	val_pred = _inv_log_price(val_pred_log)
	val_pred = np.array([ensure_positive_price(p) for p in val_pred])
	val_true = _inv_log_price(y_val)
	val_smape = smape(val_true, val_pred)
	print(f"Validation SMAPE: {val_smape * 100:.2f}%")

	# Save model
	joblib.dump(pipe, os.path.join(MODELS_DIR, 'tfidf_ridge.joblib'))
	print(f"Saved model to {os.path.join(MODELS_DIR, 'tfidf_ridge.joblib')}")

	return pipe, val_smape


def predict_test(pipe: Pipeline) -> None:
	test_csv = os.path.join(DATA_DIR, 'test.csv')
	test_df = pd.read_csv(test_csv)
	pred_log = pipe.predict(test_df)
	pred = _inv_log_price(pred_log)
	pred = np.array([ensure_positive_price(p) for p in pred])

	out = pd.DataFrame({
		'sample_id': test_df['sample_id'],
		'price': pred.astype(float)
	})
	out.to_csv(OUT_CSV, index=False)
	print(f"Wrote predictions to {OUT_CSV}")


if __name__ == '__main__':
	pipe, _ = train_and_eval()
	predict_test(pipe)
