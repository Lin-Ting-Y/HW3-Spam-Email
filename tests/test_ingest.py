import os
import csv
import json
import pandas as pd
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INGEST_PATH = ROOT / 'scripts' / 'ingest_spam.py'


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location('ingest_spam', str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_load_raw_parses_no_header(tmp_path):
    # Create a tiny headerless dataset
    raw = tmp_path / 'raw.csv'
    rows = [
        ['ham', 'hello there'],
        ['spam', 'WIN big cash now!!'],
        ['ham', 'see you soon'],
        ['spam', 'free entry to win a car'],
    ]
    with raw.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerows(rows)

    mod = load_module(INGEST_PATH)
    df = mod.load_raw(str(raw))
    assert list(df.columns) == ['label', 'text']
    assert len(df) == 4
    assert set(df['label'].unique()) == {'ham', 'spam'}


def test_split_and_save_creates_stratified_splits(tmp_path):
    # Prepare module and redirect output paths to tmp directories
    mod = load_module(INGEST_PATH)
    mod.OUT_DIR = str(tmp_path / 'processed')
    mod.TRAIN_PATH = str(Path(mod.OUT_DIR) / 'train.csv')
    mod.TEST_PATH = str(Path(mod.OUT_DIR) / 'test.csv')

    # Build a small balanced dataframe
    df = pd.DataFrame({
        'label': ['ham', 'spam'] * 50,
        'text': [f'msg {i}' for i in range(100)],
    })

    mod.split_and_save(df, test_size=0.2, random_state=123)

    assert os.path.exists(mod.TRAIN_PATH)
    assert os.path.exists(mod.TEST_PATH)

    train_df = pd.read_csv(mod.TRAIN_PATH)
    test_df = pd.read_csv(mod.TEST_PATH)

    # Sizes
    assert len(train_df) + len(test_df) == len(df)

    # Both labels present in each split (due to stratification)
    assert set(train_df['label'].unique()) == {'ham', 'spam'}
    assert set(test_df['label'].unique()) == {'ham', 'spam'}
