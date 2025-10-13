import os
import math
import pandas as pd
import numpy as np
from typing import Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import make_scorer

from utils import smape, ensure_positive_price

# Resolve dataset directory priority:
# 1) ENV DATA_DIR
# 2) student_resource/dataset
# 3) dataset
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ENV_DATA_DIR = os.environ.get('DATA_DIR')
SR_DATA_DIR = os.path.join(_REPO_ROOT, 'student_resource', 'dataset')
FALLBACK_DATA_DIR = os.path.join(_REPO_ROOT, 'dataset')

if ENV_DATA_DIR and os.path.isdir(ENV_DATA_DIR):
	DATA_DIR = ENV_DATA_DIR
elif os.path.isdir(SR_DATA_DIR):
	DATA_DIR = SR_DATA_DIR
else:
	DATA_DIR = FALLBACK_DATA_DIR

TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
OUT_CSV = os.path.abspath(os.path.join(_REPO_ROOT, 'test_out.csv'))


def _extract_text(df: pd.DataFrame) -> pd.Series:
	text = df['catalog_content'].fillna('')
	return text


def _log_price(y: np.ndarray) -> np.ndarray:
	return np.log(np.clip(y, 0.01, None))


def _inv_log_price(y_log: np.ndarray) -> np.ndarray:
	return np.exp(y_log)


def train_and_eval(random_state: int = 42) -> Tuple[Pipeline, float]:
	df = pd.read_csv(TRAIN_CSV)
	df = df.dropna(subset=['catalog_content', 'price'])

	# Target transform
	y = df['price'].astype(float).values
	y_log = _log_price(y)

	X_train, X_val, y_train, y_val = train_test_split(
		df, y_log, test_size=0.2, random_state=random_state
	)

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

	return pipe, val_smape


def predict_test(pipe: Pipeline) -> None:
	test_df = pd.read_csv(TEST_CSV)
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
