from pydantic import BaseModel
from typing import Optional


class PredictionRequest(BaseModel):
    """Input features for credit risk prediction."""
    Amount: float
    PricingStrategy: int
    total_amount: float
    mean_amount: float
    std_amount: float
    transaction_count: int
    max_amount: float
    min_amount: float
    total_value: float
    mean_value: float
    transaction_hour: int
    transaction_day: int
    transaction_month: int
    transaction_year: int
    transaction_dayofweek: int
    ProductCategory_airtime: bool = False
    ProductCategory_data_bundles: bool = False
    ProductCategory_financial_services: bool = False
    ProductCategory_movies: bool = False
    ProductCategory_other: bool = False
    ProductCategory_ticket: bool = False
    ProductCategory_transport: bool = False
    ProductCategory_tv: bool = False
    ProductCategory_utility_bill: bool = False
    ChannelId_ChannelId_1: bool = False
    ChannelId_ChannelId_2: bool = False
    ChannelId_ChannelId_3: bool = False
    ChannelId_ChannelId_5: bool = False
    ProviderId_ProviderId_1: bool = False
    ProviderId_ProviderId_2: bool = False
    ProviderId_ProviderId_3: bool = False
    ProviderId_ProviderId_4: bool = False
    ProviderId_ProviderId_5: bool = False
    ProviderId_ProviderId_6: bool = False


class PredictionResponse(BaseModel):
    """Output of the credit risk prediction."""
    customer_id: Optional[str] = None
    risk_probability: float
    is_high_risk: int
    risk_label: str
    model_version: str = "1"