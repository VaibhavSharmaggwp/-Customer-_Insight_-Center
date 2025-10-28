# utils/data_loader.py
"""
Data Loader – Load, parse dates, clean 7 CSV files
--------------------------------------------------
- Function: load_and_preprocess() → dict of DataFrames
- Direct run: prints sample data for debugging
"""

import pandas as pd
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv

# Load .env (GROQ_API_KEY, etc.)
load_dotenv()

# Path to your DATASET folder
DATASET_PATH = Path(__file__).parent.parent / "DATASET"


def load_and_preprocess() -> Dict[str, pd.DataFrame]:
    """
    Load all 7 CSV files, convert dates, clean data.
    Returns: {'orders': df, 'routes': df, ...}
    """
    print("Starting data load from:", DATASET_PATH)

    files = {
        "orders": "orders.csv",
        "routes": "routes_distance.csv",
        "delivery": "delivery_performance.csv",
        "cost": "cost_breakdown.csv",
        "feedback": "customer_feedback.csv",
        "fleet": "vehicle_fleet.csv",
        "inventory": "warehouse_inventory.csv",
    }

    data = {}  # This holds all DataFrames

    for key, filename in files.items():
        file_path = DATASET_PATH / filename

        # 1. Check file exists
        if not file_path.exists():
            raise FileNotFoundError(f"Missing: {file_path}")

        print(f"\nLoading {filename}...")

        # 2. Read CSV
        df = pd.read_csv(file_path)

        # 3. Parse known date columns
        date_cols = {
            "orders": ["Order_Date"],
            "feedback": ["Feedback_Date"],
            "inventory": ["Last_Restocked_Date"],
        }.get(key, [])

        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                print(f"  → Parsed date: {col}")

        # 4. Clean Order_ID
        if "Order_ID" in df.columns:
            df["Order_ID"] = df["Order_ID"].astype(str).str.strip()
            invalid = df["Order_ID"].isna() | (df["Order_ID"] == "")
            if invalid.any():
                print(f"  → Dropped {invalid.sum()} rows with bad Order_ID")
                df = df[~invalid].copy()

        # 5. File-specific cleaning
        if key == "delivery":
            df["Customer_Rating"] = df["Customer_Rating"].fillna(3)
        if key == "feedback":
            df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").fillna(3).clip(1, 5)
        if key == "routes":
            num_cols = ["Distance_KM", "Fuel_Consumption_L", "Toll_Charges_INR", "Traffic_Delay_Minutes"]
            for c in num_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

        # 6. Store in dict (THIS WAS THE BUG!)
        data[key] = df
        print(f"  → {len(df)} rows → stored as '{key}'")

    print("\nAll 7 datasets loaded and cleaned!")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# DEBUG: Run this file directly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        datasets = load_and_preprocess()
        print("\nSample from 'orders':")
        print(datasets["orders"].head(2))
        print("\nSample from 'delivery':")
        print(datasets["delivery"].head(2))
    except Exception as e:
        print("ERROR:", e)