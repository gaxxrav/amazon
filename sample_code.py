import os
import pandas as pd

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset'))
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
OUT_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_out.csv'))


def main():
	if not os.path.exists(TEST_CSV):
		raise FileNotFoundError(f"Missing {TEST_CSV}")
	
test_df = pd.read_csv(TEST_CSV)
	# Dummy constant prediction just for format demonstration
	out = test_df[['sample_id']].copy()
	out['price'] = 9.99
	out.to_csv(OUT_CSV, index=False)
	print(f"Wrote sample predictions to {OUT_CSV}")


if __name__ == '__main__':
	main()
