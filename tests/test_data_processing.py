import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import (
    AggregateFeatures,
    TimeFeatures,
    CategoricalEncoder,
    DropColumns,
    build_rfm,
    assign_risk_label,
    build_pipeline
)

# ── Fixtures ──────────────────────────────────────────
@pytest.fixture
def sample_df():
    """Small sample DataFrame mimicking raw Xente data."""
    return pd.DataFrame({
        'TransactionId': ['T1', 'T2', 'T3', 'T4', 'T5'],
        'BatchId': ['B1', 'B1', 'B2', 'B2', 'B3'],
        'AccountId': ['A1', 'A2', 'A1', 'A3', 'A2'],
        'SubscriptionId': ['S1', 'S2', 'S1', 'S3', 'S2'],
        'CustomerId': ['C1', 'C2', 'C1', 'C3', 'C2'],
        'CurrencyCode': ['UGX'] * 5,
        'CountryCode': [256] * 5,
        'ProviderId': ['ProviderId_1', 'ProviderId_2', 'ProviderId_1',
                       'ProviderId_3', 'ProviderId_2'],
        'ProductId': ['P1', 'P2', 'P1', 'P3', 'P2'],
        'ProductCategory': ['airtime', 'financial_services', 'airtime',
                            'utility_bill', 'financial_services'],
        'ChannelId': ['ChannelId_1', 'ChannelId_2', 'ChannelId_1',
                      'ChannelId_3', 'ChannelId_2'],
        'Amount': [1000.0, -20.0, 500.0, 20000.0, -644.0],
        'Value': [1000, 20, 500, 21800, 644],
        'TransactionStartTime': [
            '2018-11-15T02:18:49Z',
            '2018-11-15T02:19:08Z',
            '2018-11-16T10:00:00Z',
            '2018-11-17T15:30:00Z',
            '2018-11-18T08:00:00Z'
        ],
        'PricingStrategy': [2, 2, 2, 2, 2],
        'FraudResult': [0, 0, 0, 0, 1]
    })


# ── Test 1: AggregateFeatures ─────────────────────────
def test_aggregate_features_adds_columns(sample_df):
    """AggregateFeatures should add 8 new aggregate columns."""
    transformer = AggregateFeatures()
    result = transformer.fit_transform(sample_df)
    expected_cols = [
        'total_amount', 'mean_amount', 'std_amount',
        'transaction_count', 'max_amount', 'min_amount',
        'total_value', 'mean_value'
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


# ── Test 2: AggregateFeatures correct values ──────────
def test_aggregate_features_correct_values(sample_df):
    """C1 has 2 transactions with amounts 1000 and 500."""
    transformer = AggregateFeatures()
    result = transformer.fit_transform(sample_df)
    c1_rows = result[result['CustomerId'] == 'C1']
    assert c1_rows['transaction_count'].iloc[0] == 2
    assert c1_rows['total_amount'].iloc[0] == 1500.0


# ── Test 3: TimeFeatures ──────────────────────────────
def test_time_features_adds_columns(sample_df):
    """TimeFeatures should add 5 time columns."""
    transformer = TimeFeatures()
    result = transformer.fit_transform(sample_df)
    expected_cols = [
        'transaction_hour', 'transaction_day',
        'transaction_month', 'transaction_year',
        'transaction_dayofweek'
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


# ── Test 4: TimeFeatures correct values ───────────────
def test_time_features_correct_hour(sample_df):
    """First transaction is at 02:18:49 so hour should be 2."""
    transformer = TimeFeatures()
    result = transformer.fit_transform(sample_df)
    assert result['transaction_hour'].iloc[0] == 2


# ── Test 5: DropColumns removes identifiers ───────────
def test_drop_columns_removes_ids(sample_df):
    """DropColumns should remove TransactionId, BatchId, etc."""
    transformer = DropColumns()
    result = transformer.fit_transform(sample_df)
    dropped = ['TransactionId', 'BatchId', 'AccountId',
               'SubscriptionId', 'CurrencyCode', 'CountryCode']
    for col in dropped:
        assert col not in result.columns, f"Column should be dropped: {col}"


# ── Test 6: build_rfm produces correct shape ──────────
def test_build_rfm_shape(sample_df):
    """RFM should have one row per unique customer."""
    rfm = build_rfm(sample_df, snapshot_date='2019-01-01')
    assert rfm.shape[0] == sample_df['CustomerId'].nunique()
    assert 'Recency' in rfm.columns
    assert 'Frequency' in rfm.columns
    assert 'Monetary' in rfm.columns


# ── Test 7: assign_risk_label adds is_high_risk ───────
def test_assign_risk_label_binary(sample_df):
    """is_high_risk should only contain 0 and 1."""
    rfm = build_rfm(sample_df, snapshot_date='2019-01-01')
    rfm_labeled = assign_risk_label(rfm, random_state=42)
    assert 'is_high_risk' in rfm_labeled.columns
    assert set(rfm_labeled['is_high_risk'].unique()).issubset({0, 1})


# ── Test 8: pipeline runs end to end ─────────────────
def test_pipeline_runs(sample_df):
    """Full pipeline should run without errors."""
    pipeline = build_pipeline()
    result = pipeline.fit_transform(sample_df)
    assert result.shape[0] == sample_df.shape[0]
    assert 'CustomerId' in result.columns