import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = os.path.join('data', 'sms_spam_no_header.csv')
OUT_DIR = os.path.join('data', 'processed')
TRAIN_PATH = os.path.join(OUT_DIR, 'train.csv')
TEST_PATH = os.path.join(OUT_DIR, 'test.csv')


def load_raw(path: str) -> pd.DataFrame:
    # Dataset has no header: first col=label, second col=text
    df = pd.read_csv(path, header=None, names=['label', 'text'], encoding='latin-1')
    # drop rows with NaN or empty text
    df = df.dropna(subset=['label', 'text'])
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'] != '']
    return df


def split_and_save(df: pd.DataFrame, test_size: float, random_state: int) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label']
    )
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f'Saved train -> {TRAIN_PATH} ({len(train_df)})')
    print(f'Saved test  -> {TEST_PATH} ({len(test_df)})')


def main():
    parser = argparse.ArgumentParser(description='Ingest SMS spam dataset and create splits.')
    parser.add_argument('--raw', default=RAW_PATH, help='Path to raw csv (no header)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split size (0-1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    if not os.path.exists(args.raw):
        raise FileNotFoundError(f'Raw dataset not found: {args.raw}')

    df = load_raw(args.raw)
    split_and_save(df, args.test_size, args.seed)


if __name__ == '__main__':
    main()
