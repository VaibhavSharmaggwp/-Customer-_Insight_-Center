# test_loader.py
import sys
print("Python path:", sys.path)

try:
    from utils.data_loader import load_and_preprocess
    print("Import OK")
    data = load_and_preprocess()
    print(data["orders"].head(2))
except Exception as e:
    print("ERROR:", e)