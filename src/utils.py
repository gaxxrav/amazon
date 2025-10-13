import os
import time
import hashlib
from typing import List, Optional

import requests
from PIL import Image
from io import BytesIO

DEFAULT_IMAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'images'))


def _safe_mkdir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def _url_to_filename(url: str) -> str:
	# Stable deterministic filename from URL
	h = hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]
	return f"{h}.jpg"


def download_image(
	url: str,
	dest_dir: Optional[str] = None,
	retries: int = 3,
	backoff_seconds: float = 1.0,
	timeout: float = 10.0,
	convert_rgb: bool = True,
) -> Optional[str]:
	"""
	Download a single image with retries and caching.

	Returns local filepath if successful, else None.
	"""
	if not url:
		return None

	dest_dir = dest_dir or DEFAULT_IMAGE_DIR
	_safe_mkdir(dest_dir)

	filename = _url_to_filename(url)
	local_path = os.path.join(dest_dir, filename)

	# Cache hit
	if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
		return local_path

	last_exc: Optional[Exception] = None
	for attempt in range(retries):
		try:
			resp = requests.get(url, timeout=timeout)
			if resp.status_code != 200 or not resp.content:
				raise RuntimeError(f"HTTP {resp.status_code}")
			img = Image.open(BytesIO(resp.content))
			if convert_rgb and img.mode != 'RGB':
				img = img.convert('RGB')
			img.save(local_path, format='JPEG', quality=90)
			return local_path
		except Exception as exc:
			last_exc = exc
			time.sleep(backoff_seconds * (2 ** attempt))

	# Final failure, ensure no partial file remains
	try:
		if os.path.exists(local_path) and os.path.getsize(local_path) == 0:
			os.remove(local_path)
	except Exception:
		pass
	return None


def download_images(urls: List[str], dest_dir: Optional[str] = None) -> List[Optional[str]]:
	"""
	Vectorized helper: download many images, returns list of local paths or None.
	"""
	paths: List[Optional[str]] = []
	for url in urls:
		paths.append(download_image(url, dest_dir=dest_dir))
	return paths


def ensure_positive_price(value: float, minimum: float = 0.01) -> float:
	return float(max(value, minimum))


def smape(y_true, y_pred) -> float:
	"""Symmetric Mean Absolute Percentage Error (in fraction, not %)."""
	import numpy as np
	y_true = np.asarray(y_true, dtype=float)
	y_pred = np.asarray(y_pred, dtype=float)
	num = np.abs(y_true - y_pred)
	denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
	# Avoid division by zero
	mask = denom == 0
	denom[mask] = 1.0
	return float(np.mean(num / denom))
