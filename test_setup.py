# test_setup.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

ROOT = Path(__file__).parent
DATASET_PATH = ROOT / "DATASET"
DATA_PATH = ROOT / "data"

print("Setup Check:")
print(f"  Root: {ROOT}")
print(f"  DATASET exists: {DATASET_PATH.exists()}")
print(f"  data/ exists: {DATA_PATH.exists()}")

print("\nDatasets found:")
for f in sorted(DATASET_PATH.glob("*.csv")):
    print(f"  → {f.name}")

print("\nAll imports successful!")