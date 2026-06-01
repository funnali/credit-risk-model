import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AggregateFeatures(BaseEstimator, TransformerMixin):
    """Creates customer-level aggregate features from transaction data."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info("Creating aggregate features...")
        df = X.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        agg = df.groupby('CustomerId').agg(
            total_amount=('Amount', 'sum'),
            mean_amount=('Amount', 'mean'),
            std_amount=('Amount', 'std'),
            transaction_count=('TransactionId', 'count'),
            max_amount=('Amount', 'max'),
            min_amount=('Amount', 'min'),
            total_value=('Value', 'sum'),
            mean_value=('Value', 'mean'),
        ).reset_index()
        agg['std_amount'] = agg['std_amount'].fillna(0)
        df = df.merge(agg, on='CustomerId', how='left')
        logger.info(f"Aggregate features created. Shape: {df.shape}")
        return df


class TimeFeatures(BaseEstimator, TransformerMixin):
    """Extracts time-based features from TransactionStartTime."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info("Extracting time features...")
        df = X.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        df['transaction_hour'] = df['TransactionStartTime'].dt.hour
        df['transaction_day'] = df['TransactionStartTime'].dt.day
        df['transaction_month'] = df['TransactionStartTime'].dt.month
        df['transaction_year'] = df['TransactionStartTime'].dt.year
        df['transaction_dayofweek'] = df['TransactionStartTime'].dt.dayofweek
        logger.info("Time features extracted.")
        return df


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """One-hot encodes categorical columns."""

    def __init__(self):
        self.cat_columns = ['ProductCategory', 'ChannelId', 'ProviderId']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info("Encoding categorical features...")
        df = X.copy()
        df = pd.get_dummies(df, columns=self.cat_columns, drop_first=False)
        logger.info(f"Categorical encoding done. Shape: {df.shape}")
        return df


class DropColumns(BaseEstimator, TransformerMixin):
    """Drops identifier and redundant columns."""

    def __init__(self):
        self.drop_cols = [
            'TransactionId', 'BatchId', 'AccountId',
            'SubscriptionId', 'CurrencyCode', 'CountryCode',
            'TransactionStartTime', 'ProductId', 'Value'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info("Dropping unnecessary columns...")
        df = X.copy()
        cols_to_drop = [c for c in self.drop_cols if c in df.columns]
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Columns dropped. Shape: {df.shape}")
        return df


class ScaleFeatures(BaseEstimator, TransformerMixin):
    """Standardizes numerical features using StandardScaler."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.num_cols = None

    def fit(self, X, y=None):
        self.num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if 'is_high_risk' in self.num_cols:
            self.num_cols.remove('is_high_risk')
        if 'FraudResult' in self.num_cols:
            self.num_cols.remove('FraudResult')
        self.scaler.fit(X[self.num_cols])
        return self

    def transform(self, X):
        logger.info("Scaling numerical features...")
        df = X.copy()
        df[self.num_cols] = self.scaler.transform(df[self.num_cols])
        logger.info("Scaling done.")
        return df


def build_pipeline():
    """Returns the full feature engineering pipeline."""
    pipeline = Pipeline(steps=[
        ('aggregate', AggregateFeatures()),
        ('time_features', TimeFeatures()),
        ('encode', CategoricalEncoder()),
        ('drop_cols', DropColumns()),
    ])
    return pipeline


def process_data(input_path: str, output_path: str = None) -> pd.DataFrame:
    """Loads raw data, runs the pipeline, returns processed DataFrame."""
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Raw data shape: {df.shape}")
    pipeline = build_pipeline()
    df_processed = pipeline.fit_transform(df)
    logger.info(f"Processed data shape: {df_processed.shape}")
    if output_path:
        df_processed.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
    return df_processed


def build_rfm(df: pd.DataFrame, snapshot_date: str = '2019-01-01') -> pd.DataFrame:
    """Calculates RFM metrics per customer."""
    logger.info("Building RFM features...")
    df = df.copy()
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    snapshot = pd.Timestamp(snapshot_date, tz='UTC')
    rfm = df.groupby('CustomerId').agg(
        Recency=('TransactionStartTime', lambda x: (snapshot - x.max()).days),
        Frequency=('TransactionId', 'count'),
        Monetary=('Amount', lambda x: x[x > 0].sum())
    ).reset_index()
    rfm['Monetary'] = rfm['Monetary'].fillna(0)
    logger.info(f"RFM shape: {rfm.shape}")
    return rfm


def assign_risk_label(rfm: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Clusters customers into 3 groups using K-Means on RFM.
    Labels the least engaged cluster as is_high_risk = 1.
    """
    logger.info("Running K-Means clustering on RFM...")
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    cluster_summary = rfm.groupby('Cluster').agg(
        mean_recency=('Recency', 'mean'),
        mean_frequency=('Frequency', 'mean'),
        mean_monetary=('Monetary', 'mean')
    )
    logger.info(f"\nCluster summary:\n{cluster_summary}")

    cluster_summary['risk_score'] = (
        cluster_summary['mean_recency'] -
        cluster_summary['mean_frequency'] -
        cluster_summary['mean_monetary'] / 1000
    )
    high_risk_cluster = cluster_summary['risk_score'].idxmax()
    logger.info(f"High-risk cluster: Cluster {high_risk_cluster}")

    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
    logger.info(f"High-risk customers: {rfm['is_high_risk'].sum():,} "
                f"({rfm['is_high_risk'].mean()*100:.1f}%)")
    return rfm[['CustomerId', 'Recency', 'Frequency', 'Monetary',
                'Cluster', 'is_high_risk']]


if __name__ == "__main__":
    # Step 1: Feature engineering
    df_raw = pd.read_csv("data/raw/data.csv")
    df_processed = process_data(
        input_path="data/raw/data.csv",
        output_path="data/processed/processed_data.csv"
    )

    # Step 2: RFM + risk labels
    rfm = build_rfm(df_raw, snapshot_date='2019-01-01')
    rfm_labeled = assign_risk_label(rfm, random_state=42)

    # Step 3: Merge is_high_risk into processed data
    df_final = df_processed.merge(
        rfm_labeled[['CustomerId', 'is_high_risk']],
        on='CustomerId',
        how='left'
    )
    df_final['is_high_risk'] = df_final['is_high_risk'].fillna(0).astype(int)
    df_final.to_csv("data/processed/final_data.csv", index=False)

    print(f"\nFinal dataset shape: {df_final.shape}")
    print(f"High-risk rows:  {df_final['is_high_risk'].sum():,}")
    print(f"Low-risk rows:   {(df_final['is_high_risk']==0).sum():,}")
    print(f"\nSample RFM clusters:\n{rfm_labeled.head(10)}")