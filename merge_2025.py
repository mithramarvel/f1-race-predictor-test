import pandas as pd

qualifying_df = pd.read_csv("f1_2025_quali_data.csv")
race_df = pd.read_csv("f1_2025_race_data.csv")

qualifying_df['Driver'] = qualifying_df['Driver'].str.strip()
race_df['Driver'] = race_df['Driver'].str.strip()

qualifying_df['Race'] = qualifying_df['Race'].str.strip()
race_df['Race'] = race_df['Race'].str.strip()

merged_df = pd.merge(qualifying_df, race_df, on=['Race', 'Driver'], how='inner', suffixes=('_Quali', '_Race'))

merged_df.to_csv("f1_2025_combined_data.csv", index=False)
print("✅ Saved merged dataset to f1_2025_combined_data.csv")