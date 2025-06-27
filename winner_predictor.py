import pandas as pd
import joblib

# === Load Trained Model & Encoders ===
model = joblib.load('models/winner_predictor.pkl')
driver_encoder = joblib.load('models/driver_encoder.pkl')
team_encoder = joblib.load('models/team_encoder.pkl')
race_encoder = joblib.load('models/race_encoder.pkl')

# === Load Merged Dataset ===
df = pd.read_csv('f1_2025_quali_data.csv')


# === User Input: Race to Predict ===
selected_race = '2025 Canadian Grand Prix'


# === Filter Data for Selected Race ===
race_df = df[df['Race'] == selected_race].copy()
if race_df.empty:
    print(f"❌ No data found for race: {selected_race}")
    exit()
race_df['Q1_sec'] = pd.to_numeric(race_df['Q1_sec'], errors='coerce')
race_df['Q2_sec'] = pd.to_numeric(race_df['Q2_sec'], errors='coerce')
race_df['Q3_sec'] = pd.to_numeric(race_df['Q3_sec'], errors='coerce')
race_df = race_df.dropna(subset=['Q1_sec', 'Q2_sec', 'Q3_sec'])
race_df['Avg_Quali_Time'] = race_df[['Q1_sec', 'Q2_sec', 'Q3_sec']].mean(axis=1)
race_df['Grid_Pos'] = pd.to_numeric(race_df['Grid_Pos'], errors='coerce')
race_df = race_df.dropna(subset=['Avg_Quali_Time', 'Grid_Pos', 'Driver', 'Team'])


# === Encode Categorical Variables ===

# Filter out unseen teams and drivers
race_df = race_df[
    race_df['Team'].isin(team_encoder.classes_) &
    race_df['Driver'].isin(driver_encoder.classes_)
]

# Only proceed if we still have data left
if race_df.empty:
    print(f"❌ No valid drivers or teams found for race: {selected_race}")
    exit()

# Encode
race_df['Team_encoded'] = team_encoder.transform(race_df['Team'])
race_df['Driver_encoded'] = driver_encoder.transform(race_df['Driver'])

race_df['Race_encoded'] = race_encoder.transform(race_df['Race'])

# === Features for Prediction ===
features = ['Avg_Quali_Time', 'Grid_Pos', 'Team_encoded', 'Driver_encoded', 'Race_encoded']

# === Make Predictions ===
race_df['Predicted_Winner'] = model.predict(race_df[features])
race_df['Win_Probability'] = model.predict_proba(race_df[features])[:, 1]

# === Sort by Probability ===

race_df_sorted = race_df.sort_values(by='Win_Probability', ascending=False)

# === Show Results ===
print(f"\n🏁 Predictions for {selected_race}\n")
print(race_df_sorted[['Driver', 'Team', 'Grid_Pos', 'Avg_Quali_Time', 'Win_Probability', 'Predicted_Winner']].to_string(index=False))
