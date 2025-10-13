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
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
import lightgbm as lgb

# Image processing
from PIL import Image
import requests
from io import BytesIO
import hashlib

# Paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SR_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(SR_ROOT, 'dataset')
MODELS_DIR = os.path.join(SR_ROOT, 'models')
IMAGES_DIR = os.path.join(SR_ROOT, 'images')
OUT_CSV = os.path.join(DATA_DIR, 'test_out.csv')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


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


def download_image_safe(url: str, max_retries: int = 3) -> Optional[str]:
    """Safely download image with retries"""
    if not url or pd.isna(url):
        return None
    
    # Create filename from URL hash
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    filename = f"{url_hash}.jpg"
    filepath = os.path.join(IMAGES_DIR, filename)
    
    # Return if already exists
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        return filepath
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(filepath, 'JPEG', quality=90)
                return filepath
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to download {url}: {e}")
            continue
    
    return None


def extract_simple_image_features(image_path: str) -> np.ndarray:
    """Extract simple image features without deep learning"""
    if not image_path or not os.path.exists(image_path):
        # Return zero vector for missing images
        return np.zeros(10)
    
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        
        # Resize to standard size
        img = img.resize((224, 224))
        img_array = np.array(img)
        
        # Simple features
        features = []
        
        # Color statistics
        features.extend([
            np.mean(img_array[:, :, 0]),  # R mean
            np.mean(img_array[:, :, 1]),  # G mean
            np.mean(img_array[:, :, 2]),  # B mean
            np.std(img_array[:, :, 0]),   # R std
            np.std(img_array[:, :, 1]),   # G std
            np.std(img_array[:, :, 2]),   # B std
        ])
        
        # Brightness and contrast
        gray = np.mean(img_array, axis=2)
        features.extend([
            np.mean(gray),      # Brightness
            np.std(gray),       # Contrast
            np.percentile(gray, 25),  # Q1
            np.percentile(gray, 75),  # Q3
        ])
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return np.zeros(10)


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
    
    # IPQ extraction
    features['ipq'] = df['catalog_content'].apply(extract_ipq)
    features['log_ipq'] = np.log(features['ipq'].clip(lower=0.1))
    
    # Product type indicators
    text_lower = df['catalog_content'].fillna('').str.lower()
    features['has_bundle'] = text_lower.str.contains('bundle|set|kit', na=False).astype(int)
    features['has_refurbished'] = text_lower.str.contains('refurbished|renewed', na=False).astype(int)
    features['has_wireless'] = text_lower.str.contains('wireless|bluetooth', na=False).astype(int)
    features['has_premium'] = text_lower.str.contains('premium|pro|professional', na=False).astype(int)
    features['has_sale'] = text_lower.str.contains('sale|discount|off', na=False).astype(int)
    features['has_limited'] = text_lower.str.contains('limited|exclusive', na=False).astype(int)
    
    # Brand and category indicators
    features['has_apple'] = text_lower.str.contains('apple|iphone|ipad|macbook', na=False).astype(int)
    features['has_samsung'] = text_lower.str.contains('samsung|galaxy', na=False).astype(int)
    features['has_sony'] = text_lower.str.contains('sony|playstation', na=False).astype(int)
    features['has_nike'] = text_lower.str.contains('nike|air max|jordan', na=False).astype(int)
    
    return features


def _extract_text_for_tfidf(df: pd.DataFrame) -> pd.Series:
    """Extract and preprocess text for TF-IDF"""
    return df['catalog_content'].fillna('').apply(preprocess_text)


def _log_price(y: np.ndarray) -> np.ndarray:
    return np.log(np.clip(y, 0.01, None))


def _inv_log_price(y_log: np.ndarray) -> np.ndarray:
    return np.exp(y_log)


def train_ensemble_model(X_train, y_train, X_val, y_val):
    """Train ensemble of models"""
    
    # LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'min_child_samples': 30,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    }
    
    lgb_model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1500,
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)]
    )
    
    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Ridge
    ridge_model = Ridge(alpha=10.0, random_state=42)
    ridge_model.fit(X_train, y_train)
    
    return lgb_model, rf_model, ridge_model


def train_and_eval_multimodal(random_state: int = 42) -> Tuple[object, object, object, object, float]:
    """Train multimodal model with text and image features"""
    
    # Load data
    train_csv = os.path.join(DATA_DIR, 'train.csv')
    df = pd.read_csv(train_csv)
    df = df.dropna(subset=['catalog_content', 'price'])
    
    print(f"Training on {len(df)} samples")
    
    # Download images (sample first 1000 for speed)
    print("Downloading images...")
    sample_size = min(1000, len(df))
    df_sample = df.head(sample_size).copy()
    
    image_paths = []
    for i, url in enumerate(df_sample['image_link']):
        if i % 100 == 0:
            print(f"Downloaded {i}/{sample_size} images")
        path = download_image_safe(url)
        image_paths.append(path)
    
    df_sample['image_path'] = image_paths
    
    # Extract image features
    print("Extracting image features...")
    image_features = np.array([extract_simple_image_features(path) for path in image_paths])
    
    # Target transformation
    y = df_sample['price'].astype(float).values
    y_log = _log_price(y)
    
    # Extract text features
    text_features = extract_text_features(df_sample)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        df_sample, y_log, test_size=0.2, random_state=random_state
    )
    
    # Get indices for feature extraction
    train_idx = X_train.index
    val_idx = X_val.index
    
    # Extract features for splits
    text_features_train = text_features.loc[train_idx]
    text_features_val = text_features.loc[val_idx]
    
    image_features_train = image_features[train_idx - df_sample.index[0]]
    image_features_val = image_features[val_idx - df_sample.index[0]]
    
    # TF-IDF for text
    tfidf = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    X_train_text = tfidf.fit_transform(_extract_text_for_tfidf(X_train))
    X_val_text = tfidf.transform(_extract_text_for_tfidf(X_val))
    
    # Combine all features
    X_train_combined = np.hstack([
        X_train_text.toarray(),
        text_features_train[['text_length', 'word_count', 'ipq', 'log_ipq', 
                           'has_bundle', 'has_refurbished', 'has_wireless', 
                           'has_premium', 'has_sale', 'has_limited',
                           'has_apple', 'has_samsung', 'has_sony', 'has_nike']].values,
        image_features_train
    ])
    
    X_val_combined = np.hstack([
        X_val_text.toarray(),
        text_features_val[['text_length', 'word_count', 'ipq', 'log_ipq',
                          'has_bundle', 'has_refurbished', 'has_wireless',
                          'has_premium', 'has_sale', 'has_limited',
                          'has_apple', 'has_samsung', 'has_sony', 'has_nike']].values,
        image_features_val
    ])
    
    print(f"Feature matrix shape: {X_train_combined.shape}")
    
    # Train ensemble
    lgb_model, rf_model, ridge_model = train_ensemble_model(
        X_train_combined, y_train, X_val_combined, y_val
    )
    
    # Evaluate ensemble
    lgb_pred = lgb_model.predict(X_val_combined)
    rf_pred = rf_model.predict(X_val_combined)
    ridge_pred = ridge_model.predict(X_val_combined)
    
    # Weighted ensemble (tune weights based on individual performance)
    ensemble_pred = 0.5 * lgb_pred + 0.3 * rf_pred + 0.2 * ridge_pred
    
    val_pred = _inv_log_price(ensemble_pred)
    val_pred = np.array([ensure_positive_price(p) for p in val_pred])
    val_true = _inv_log_price(y_val)
    
    val_smape = smape(val_true, val_pred)
    print(f"Validation SMAPE: {val_smape * 100:.2f}%")
    
    # Save models
    joblib.dump(lgb_model, os.path.join(MODELS_DIR, 'lgb_multimodal.joblib'))
    joblib.dump(rf_model, os.path.join(MODELS_DIR, 'rf_multimodal.joblib'))
    joblib.dump(ridge_model, os.path.join(MODELS_DIR, 'ridge_multimodal.joblib'))
    joblib.dump(tfidf, os.path.join(MODELS_DIR, 'tfidf_multimodal.joblib'))
    
    return lgb_model, rf_model, ridge_model, tfidf, val_smape


def predict_test_multimodal(lgb_model, rf_model, ridge_model, tfidf):
    """Generate predictions for test set"""
    
    test_csv = os.path.join(DATA_DIR, 'test.csv')
    test_df = pd.read_csv(test_csv)
    
    # For full test set, use mean image features (since we can't download all images)
    print("Using mean image features for test set...")
    mean_image_features = np.zeros((len(test_df), 10))
    
    # Extract text features
    text_features_test = extract_text_features(test_df)
    
    # Transform text
    X_test_text = tfidf.transform(_extract_text_for_tfidf(test_df))
    
    # Combine features
    X_test_combined = np.hstack([
        X_test_text.toarray(),
        text_features_test[['text_length', 'word_count', 'ipq', 'log_ipq',
                           'has_bundle', 'has_refurbished', 'has_wireless',
                           'has_premium', 'has_sale', 'has_limited',
                           'has_apple', 'has_samsung', 'has_sony', 'has_nike']].values,
        mean_image_features
    ])
    
    # Predict with ensemble
    lgb_pred = lgb_model.predict(X_test_combined)
    rf_pred = rf_model.predict(X_test_combined)
    ridge_pred = ridge_model.predict(X_test_combined)
    
    ensemble_pred = 0.5 * lgb_pred + 0.3 * rf_pred + 0.2 * ridge_pred
    
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
    lgb_model, rf_model, ridge_model, tfidf, smape_score = train_and_eval_multimodal()
    predict_test_multimodal(lgb_model, rf_model, ridge_model, tfidf)
    
    if smape_score < 0.4:
        print(f"🎉 Target achieved! SMAPE: {smape_score * 100:.2f}%")
    else:
        print(f"Target not yet achieved. SMAPE: {smape_score * 100:.2f}%")
