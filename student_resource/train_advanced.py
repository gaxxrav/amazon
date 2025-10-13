import os
import re
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

# Paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SR_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(SR_ROOT, 'dataset')
MODELS_DIR = os.path.join(SR_ROOT, 'models')
OUT_CSV = os.path.join(DATA_DIR, 'test_out.csv')

os.makedirs(MODELS_DIR, exist_ok=True)


def smape(y_true, y_pred) -> float:
    """Symmetric Mean Absolute Percentage Error"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = np.abs(y_true - y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom == 0
    denom[mask] = 1.0
    return float(np.mean(num / denom))


def smape_lgb(y_pred, y_true):
    """SMAPE for LightGBM"""
    y_true = y_true.get_label()
    return 'smape', smape(y_true, y_pred), False


def ensure_positive_price(value: float, minimum: float = 0.01) -> float:
    return float(max(value, minimum))


def extract_ipq(text: str) -> float:
    """Extract Item Pack Quantity from text"""
    if not isinstance(text, str):
        return 1.0
    
    patterns = [
        r"pack of\s*(\d+)",
        r"(\d+)\s*pack",
        r"\b(\d+)\s*(?:pcs|pieces|counts?|ct)\b",
        r"x\s*(\d+)\b",
        r"\b(\d+)[- ]?count\b",
        r"(\d+)\s*unit",
        r"(\d+)\s*item",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except:
                continue
    return 1.0


def extract_brand(text: str) -> str:
    """Extract brand from text"""
    if not isinstance(text, str):
        return "unknown"
    
    # Common brand patterns
    brand_patterns = [
        r"by\s+([A-Z][a-zA-Z0-9\s&]+?)(?:\s|$|,|\.)",
        r"from\s+([A-Z][a-zA-Z0-9\s&]+?)(?:\s|$|,|\.)",
        r"^([A-Z][a-zA-Z0-9\s&]+?)(?:\s|$|,|\.)",
    ]
    
    for pattern in brand_patterns:
        match = re.search(pattern, text)
        if match:
            brand = match.group(1).strip()
            if len(brand) > 2 and len(brand) < 50:
                return brand.lower()
    
    return "unknown"


def preprocess_text(text: str) -> str:
    """Advanced text preprocessing"""
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Preserve important units and numbers
    text = re.sub(r'(\d+)\s*(oz|ml|gb|tb|inch|ft|cm|mm|kg|lb)', r'\1\2', text, flags=re.IGNORECASE)
    
    # Clean up
    text = text.strip().lower()
    
    return text


def extract_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract comprehensive text features"""
    features = pd.DataFrame(index=df.index)
    
    # Basic text features
    features['text_length'] = df['catalog_content'].fillna('').str.len()
    features['word_count'] = df['catalog_content'].fillna('').str.split().str.len()
    features['char_count'] = df['catalog_content'].fillna('').str.len()
    
    # IPQ extraction
    features['ipq'] = df['catalog_content'].apply(extract_ipq)
    features['log_ipq'] = np.log(features['ipq'].clip(lower=0.1))
    
    # Brand extraction
    features['brand'] = df['catalog_content'].apply(extract_brand)
    
    # Product type indicators
    text_lower = df['catalog_content'].fillna('').str.lower()
    features['has_bundle'] = text_lower.str.contains('bundle|set|kit', na=False).astype(int)
    features['has_refurbished'] = text_lower.str.contains('refurbished|renewed', na=False).astype(int)
    features['has_wireless'] = text_lower.str.contains('wireless|bluetooth', na=False).astype(int)
    features['has_premium'] = text_lower.str.contains('premium|pro|professional', na=False).astype(int)
    
    # Price indicators
    features['has_sale'] = text_lower.str.contains('sale|discount|off', na=False).astype(int)
    features['has_limited'] = text_lower.str.contains('limited|exclusive', na=False).astype(int)
    
    return features


def _extract_text_for_tfidf(df: pd.DataFrame) -> pd.Series:
    """Extract and preprocess text for TF-IDF"""
    return df['catalog_content'].fillna('').apply(preprocess_text)


def _log_price(y: np.ndarray) -> np.ndarray:
    return np.log(np.clip(y, 0.01, None))


def _inv_log_price(y_log: np.ndarray) -> np.ndarray:
    return np.exp(y_log)


def train_lightgbm_model(X_train, y_train, X_val, y_val):
    """Train LightGBM with SMAPE optimization"""
    
    # Prepare data
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Parameters optimized for SMAPE
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    }
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    return model


def train_and_eval_advanced(random_state: int = 42) -> Tuple[object, float]:
    """Train advanced model with multiple features"""
    
    # Load data
    train_csv = os.path.join(DATA_DIR, 'train.csv')
    df = pd.read_csv(train_csv)
    df = df.dropna(subset=['catalog_content', 'price'])
    
    print(f"Training on {len(df)} samples")
    
    # Target transformation
    y = df['price'].astype(float).values
    y_log = _log_price(y)
    
    # Extract text features
    text_features = extract_text_features(df)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        df, y_log, test_size=0.2, random_state=random_state
    )
    
    # Extract text features for splits
    text_features_train = text_features.loc[X_train.index]
    text_features_val = text_features.loc[X_val.index]
    
    # TF-IDF for text
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        min_df=3,
        max_df=0.95,
        stop_words='english'
    )
    
    X_train_text = tfidf.fit_transform(_extract_text_for_tfidf(X_train))
    X_val_text = tfidf.transform(_extract_text_for_tfidf(X_val))
    
    # Combine features
    X_train_combined = np.hstack([
        X_train_text.toarray(),
        text_features_train[['text_length', 'word_count', 'ipq', 'log_ipq', 
                           'has_bundle', 'has_refurbished', 'has_wireless', 
                           'has_premium', 'has_sale', 'has_limited']].values
    ])
    
    X_val_combined = np.hstack([
        X_val_text.toarray(),
        text_features_val[['text_length', 'word_count', 'ipq', 'log_ipq',
                          'has_bundle', 'has_refurbished', 'has_wireless',
                          'has_premium', 'has_sale', 'has_limited']].values
    ])
    
    print(f"Feature matrix shape: {X_train_combined.shape}")
    
    # Train LightGBM
    model = train_lightgbm_model(X_train_combined, y_train, X_val_combined, y_val)
    
    # Evaluate
    val_pred_log = model.predict(X_val_combined)
    val_pred = _inv_log_price(val_pred_log)
    val_pred = np.array([ensure_positive_price(p) for p in val_pred])
    val_true = _inv_log_price(y_val)
    
    val_smape = smape(val_true, val_pred)
    print(f"Validation SMAPE: {val_smape * 100:.2f}%")
    
    # Save model and vectorizer
    joblib.dump(model, os.path.join(MODELS_DIR, 'lightgbm_advanced.joblib'))
    joblib.dump(tfidf, os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib'))
    
    return model, tfidf, val_smape


def predict_test_advanced(model, tfidf):
    """Generate predictions for test set"""
    
    test_csv = os.path.join(DATA_DIR, 'test.csv')
    test_df = pd.read_csv(test_csv)
    
    # Extract text features
    text_features_test = extract_text_features(test_df)
    
    # Transform text
    X_test_text = tfidf.transform(_extract_text_for_tfidf(test_df))
    
    # Combine features
    X_test_combined = np.hstack([
        X_test_text.toarray(),
        text_features_test[['text_length', 'word_count', 'ipq', 'log_ipq',
                           'has_bundle', 'has_refurbished', 'has_wireless',
                           'has_premium', 'has_sale', 'has_limited']].values
    ])
    
    # Predict
    pred_log = model.predict(X_test_combined)
    pred = _inv_log_price(pred_log)
    pred = np.array([ensure_positive_price(p) for p in pred])
    
    # Save predictions
    out = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': pred.astype(float)
    })
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote predictions to {OUT_CSV}")


if __name__ == '__main__':
    model, tfidf, smape_score = train_and_eval_advanced()
    predict_test_advanced(model, tfidf)
    
    if smape_score < 0.4:
        print(f"🎉 Target achieved! SMAPE: {smape_score * 100:.2f}%")
    else:
        print(f"Target not yet achieved. SMAPE: {smape_score * 100:.2f}%")
