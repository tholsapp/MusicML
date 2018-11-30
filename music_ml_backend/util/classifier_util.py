import logging
import joblib
import pandas as pd

log = logging.getLogger(__name__)


def read_features(path):
    log.info(f"Reading from {path}")
    return pd.read_csv(path, index_col=False)


def save_features(path, feature_df):
    log.info(f"Saving CSV file {path!r}")
    feature_df.to_csv(path, index=False)

