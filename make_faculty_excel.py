import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/physionet-2019/training_setA")
OUTPUT = Path("XAutoNet_Faculty_Dataset_Sample.xlsx")

FEATURES = [
    "BaseExcess", "Temp", "Chloride", "Hct", "Hgb",
    "Resp", "HCO3", "SIRS", "Potassium", "Creatinine",
    "Phosphate", "FiO2", "O2Sat", "Magnesium", "SaO2",
    "Lactate", "pH", "Calcium", "BUN"
]

files = sorted(DATA_DIR.glob("*.psv"))[:20]

rows = []

for file in files:
    df = pd.read_csv(file, sep="|")

    available = [c for c in FEATURES if c in df.columns]

    for hour, row in df.iterrows():
        record = {
            "Patient_ID": file.stem,
            "Hour": hour + 1
        }

        for feature in FEATURES:
            record[feature] = row.get(feature, None)

        record["SepsisLabel"] = row.get("SepsisLabel", None)
        rows.append(record)

result = pd.DataFrame(rows)

with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
    result.to_excel(writer, sheet_name="Dataset Sample", index=False)

print(f"Created: {OUTPUT}")
print(f"Patients included: {len(files)}")
print(f"Rows: {len(result)}")
print(f"Columns: {len(result.columns)}")