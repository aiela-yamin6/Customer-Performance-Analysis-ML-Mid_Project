"""Shared paths and settings for the lab."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "customer_performance.csv"
MODEL_DIR = ROOT / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.30
SPEND_COLUMNS = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
PURCHASE_COLUMNS = ["NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
CAMPAIGN_COLUMNS = [f"AcceptedCmp{i}" for i in range(1, 6)]
