import os
import re
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, HuberRegressor, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb

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
        r"(\d+)\s*pack\s*of",
        r"set\s*of\s*(\d+)",
        r"(\d+)\s*in\s*1",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1))
                if 1 <= val <= 1000:  # Reasonable range
                    return val
            except:
                continue
    return 1.0


def extract_price_indicators(text: str) -> dict:
    """Extract price-related indicators from text"""
    if not isinstance(text, str):
        return {}
    
    text_lower = text.lower()
    
    indicators = {
        'has_bundle': int(re.search(r'bundle|set|kit|combo', text_lower) is not None),
        'has_refurbished': int(re.search(r'refurbished|renewed|used|pre-owned', text_lower) is not None),
        'has_wireless': int(re.search(r'wireless|bluetooth|wifi', text_lower) is not None),
        'has_premium': int(re.search(r'premium|pro|professional|deluxe|luxury', text_lower) is not None),
        'has_sale': int(re.search(r'sale|discount|off|clearance', text_lower) is not None),
        'has_limited': int(re.search(r'limited|exclusive|special edition', text_lower) is not None),
        'has_new': int(re.search(r'new|latest|2024|2023', text_lower) is not None),
        'has_original': int(re.search(r'original|authentic|genuine', text_lower) is not None),
        'has_warranty': int(re.search(r'warranty|guarantee', text_lower) is not None),
        'has_free_shipping': int(re.search(r'free shipping|free delivery', text_lower) is not None),
    }
    
    return indicators


def extract_brand_category(text: str) -> dict:
    """Extract brand and category indicators"""
    if not isinstance(text, str):
        return {}
    
    text_lower = text.lower()
    
    brands = {
        'has_apple': int(re.search(r'apple|iphone|ipad|macbook|airpods', text_lower) is not None),
        'has_samsung': int(re.search(r'samsung|galaxy|note', text_lower) is not None),
        'has_sony': int(re.search(r'sony|playstation|ps5|ps4', text_lower) is not None),
        'has_nike': int(re.search(r'nike|air max|jordan', text_lower) is not None),
        'has_adidas': int(re.search(r'adidas|yeezy', text_lower) is not None),
        'has_amazon': int(re.search(r'amazon|echo|fire|kindle', text_lower) is not None),
        'has_google': int(re.search(r'google|pixel|nest', text_lower) is not None),
        'has_microsoft': int(re.search(r'microsoft|xbox|surface', text_lower) is not None),
    }
    
    categories = {
        'is_electronics': int(re.search(r'phone|tablet|laptop|computer|headphone|speaker|camera|tv|monitor', text_lower) is not None),
        'is_clothing': int(re.search(r'shirt|pants|dress|shoes|jacket|hat|sock', text_lower) is not None),
        'is_home': int(re.search(r'furniture|bed|chair|table|lamp|decor', text_lower) is not None),
        'is_beauty': int(re.search(r'makeup|skincare|shampoo|perfume|cosmetic', text_lower) is not None),
        'is_sports': int(re.search(r'sport|fitness|gym|exercise|running', text_lower) is not None),
        'is_kitchen': int(re.search(r'kitchen|cook|food|drink|utensil', text_lower) is not None),
    }
    
    return {**brands, **categories}


def preprocess_text(text: str) -> str:
    """Advanced text preprocessing"""
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Preserve important units and numbers
    text = re.sub(r'(\d+)\s*(oz|ml|gb|tb|inch|ft|cm|mm|kg|lb|pound)', r'\1\2', text, flags=re.IGNORECASE)
    
    # Clean up
    text = text.strip().lower()
    
    return text


def extract_comprehensive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract comprehensive features from text"""
    features = pd.DataFrame(index=df.index)
    
    # Basic text features
    features['text_length'] = df['catalog_content'].fillna('').str.len()
    features['word_count'] = df['catalog_content'].fillna('').str.split().str.len()
    features['sentence_count'] = df['catalog_content'].fillna('').str.count(r'[.!?]+')
    features['avg_word_length'] = features['text_length'] / (features['word_count'] + 1)
    
    # IPQ extraction
    features['ipq'] = df['catalog_content'].apply(extract_ipq)
    features['log_ipq'] = np.log(features['ipq'].clip(lower=0.1))
    features['sqrt_ipq'] = np.sqrt(features['ipq'])
    
    # Price indicators
    price_indicators = df['catalog_content'].apply(extract_price_indicators)
    for key in price_indicators.iloc[0].keys():
        features[f'price_{key}'] = [d[key] for d in price_indicators]
    
    # Brand and category indicators
    brand_cat = df['catalog_content'].apply(extract_brand_category)
    for key in brand_cat.iloc[0].keys():
        features[f'brand_{key}'] = [d[key] for d in brand_cat]
    
    # Text complexity
    text_lower = df['catalog_content'].fillna('').str.lower()
    features['has_numbers'] = text_lower.str.contains(r'\d', na=False).astype(int)
    features['has_special_chars'] = text_lower.str.contains(r'[!@#$%^&*()]', na=False).astype(int)
    features['has_uppercase'] = df['catalog_content'].str.contains(r'[A-Z]', na=False).astype(int)
    
    return features


def _extract_text_for_tfidf(df: pd.DataFrame) -> pd.Series:
    """Extract and preprocess text for TF-IDF"""
    return df['catalog_content'].fillna('').apply(preprocess_text)


def _log_price(y: np.ndarray) -> np.ndarray:
    return np.log(np.clip(y, 0.01, None))


def _inv_log_price(y_log: np.ndarray) -> np.ndarray:
    return np.exp(y_log)


def train_optimized_lightgbm(X_train, y_train, X_val, y_val):
    """Train optimized LightGBM"""
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Optimized parameters for SMAPE
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'learning_rate': 0.02,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'min_child_samples': 50,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'max_depth': 12,
        'min_split_gain': 0.1,
    }
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)]
    )
    
    return model


def train_optimized_xgboost(X_train, y_train, X_val, y_val):
    """Train optimized XGBoost"""
    
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 8,
        'learning_rate': 0.03,
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'random_state': 42,
        'n_jobs': -1,
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    return model


def train_and_eval_optimized(random_state: int = 42) -> Tuple[object, object, object, float]:
    """Train optimized ensemble model"""
    
    # Load data
    train_csv = os.path.join(DATA_DIR, 'train.csv')
    df = pd.read_csv(train_csv)
    df = df.dropna(subset=['catalog_content', 'price'])
    
    print(f"Training on {len(df)} samples")
    
    # Target transformation
    y = df['price'].astype(float).values
    y_log = _log_price(y)
    
    # Extract comprehensive features
    text_features = extract_comprehensive_features(df)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        df, y_log, test_size=0.2, random_state=random_state
    )
    
    # Extract features for splits
    text_features_train = text_features.loc[X_train.index]
    text_features_val = text_features.loc[X_val.index]
    
    # Enhanced TF-IDF
    tfidf = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 4),
        min_df=2,
        max_df=0.9,
        stop_words='english',
        sublinear_tf=True,
        use_idf=True,
    )
    
    X_train_text = tfidf.fit_transform(_extract_text_for_tfidf(X_train))
    X_val_text = tfidf.transform(_extract_text_for_tfidf(X_val))
    
    # Combine all features
    feature_cols = [col for col in text_features.columns if col not in ['brand_brand', 'brand_category']]
    X_train_combined = np.hstack([
        X_train_text.toarray(),
        text_features_train[feature_cols].values
    ])
    
    X_val_combined = np.hstack([
        X_val_text.toarray(),
        text_features_val[feature_cols].values
    ])
    
    print(f"Feature matrix shape: {X_train_combined.shape}")
    
    # Train models
    print("Training LightGBM...")
    lgb_model = train_optimized_lightgbm(X_train_combined, y_train, X_val_combined, y_val)
    
    print("Training XGBoost...")
    xgb_model = train_optimized_xgboost(X_train_combined, y_train, X_val_combined, y_val)
    
    # Train Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_combined, y_train)
    
    # Evaluate ensemble
    lgb_pred = lgb_model.predict(X_val_combined)
    xgb_pred = xgb_model.predict(X_val_combined)
    rf_pred = rf_model.predict(X_val_combined)
    
    # Optimized ensemble weights
    ensemble_pred = 0.5 * lgb_pred + 0.3 * xgb_pred + 0.2 * rf_pred
    
    val_pred = _inv_log_price(ensemble_pred)
    val_pred = np.array([ensure_positive_price(p) for p in val_pred])
    val_true = _inv_log_price(y_val)
    
    val_smape = smape(val_true, val_pred)
    print(f"Validation SMAPE: {val_smape * 100:.2f}%")
    
    # Save models
    joblib.dump(lgb_model, os.path.join(MODELS_DIR, 'lgb_optimized.joblib'))
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgb_optimized.joblib'))
    joblib.dump(rf_model, os.path.join(MODELS_DIR, 'rf_optimized.joblib'))
    joblib.dump(tfidf, os.path.join(MODELS_DIR, 'tfidf_optimized.joblib'))
    
    return lgb_model, xgb_model, rf_model, tfidf, val_smape


def predict_test_optimized(lgb_model, xgb_model, rf_model, tfidf):
    """Generate predictions for test set"""
    
    test_csv = os.path.join(DATA_DIR, 'test.csv')
    test_df = pd.read_csv(test_csv)
    
    # Extract features
    text_features_test = extract_comprehensive_features(test_df)
    
    # Transform text
    X_test_text = tfidf.transform(_extract_text_for_tfidf(test_df))
    
    # Combine features
    feature_cols = [col for col in text_features_test.columns if col not in ['brand_brand', 'brand_category']]
    X_test_combined = np.hstack([
        X_test_text.toarray(),
        text_features_test[feature_cols].values
    ])
    
    # Predict with ensemble
    lgb_pred = lgb_model.predict(X_test_combined)
    xgb_pred = xgb_model.predict(X_test_combined)
    rf_pred = rf_model.predict(X_test_combined)
    
    ensemble_pred = 0.5 * lgb_pred + 0.3 * xgb_pred + 0.2 * rf_pred
    
    pred = _inv_log_price(ensemble_pred)
    pred = np.array([ensure_positive_price(p) for p in pred])
    
    # Save predictions
    out = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': pred.astype(float)
    })
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote predictions to {OUT_CSV}")


if __name__ == '__main__':
    lgb_model, xgb_model, rf_model, tfidf, smape_score = train_and_eval_optimized()
    predict_test_optimized(lgb_model, xgb_model, rf_model, tfidf)
    
    if smape_score < 0.4:
        print(f"🎉 Target achieved! SMAPE: {smape_score * 100:.2f}%")
    else:
        print(f"Target not yet achieved. SMAPE: {smape_score * 100:.2f}%")
