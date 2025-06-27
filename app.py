from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

# Initialize Flask app
app = Flask(__name__)


# === Load model and encoders ===
model = joblib.load('models/winner_predictor.pkl')
driver_encoder = joblib.load('models/driver_encoder.pkl')
team_encoder = joblib.load('models/team_encoder.pkl')
race_encoder = joblib.load('models/race_encoder.pkl')

# === Load dataset ===
df = pd.read_csv('f1_2025_combined_data.csv')

@app.route('/')
def home():
    races = sorted(df['Race'].unique())
    return render_template('index.html', races=races)

@app.route('/predict', methods=['POST'])
def predict():
    selected_race = request.form['race']
    race_df = df[df['Race'] == selected_race].copy()

    if race_df.empty:
        return render_template('result.html', error=f"No data found for {selected_race}")

    # Convert quali times
    for col in ['Q1_sec', 'Q2_sec', 'Q3_sec']:
        race_df[col] = pd.to_numeric(race_df[col], errors='coerce')
    race_df['Avg_Quali_Time'] = race_df[['Q1_sec', 'Q2_sec', 'Q3_sec']].mean(axis=1)
    race_df['Grid_Pos'] = pd.to_numeric(race_df['Grid_Pos'], errors='coerce')

    # Drop invalid rows
    race_df.dropna(subset=['Avg_Quali_Time', 'Grid_Pos', 'Driver', 'Team_Race'], inplace=True)

    # Filter valid drivers/teams only
    race_df = race_df[
        race_df['Driver'].isin(driver_encoder.classes_) &
        race_df['Team_Race'].isin(team_encoder.classes_)
    ]

    if race_df.empty:
        return render_template('result.html', error=f"No valid drivers/teams available for prediction.")

    # Encode
    race_df['Driver_encoded'] = driver_encoder.transform(race_df['Driver'])
    race_df['Team_encoded'] = team_encoder.transform(race_df['Team_Race'])
    race_df['Race_encoded'] = race_encoder.transform(race_df['Race'])

    # Select features
    features = ['Avg_Quali_Time', 'Grid_Pos', 'Team_encoded', 'Driver_encoded', 'Race_encoded']
    race_df['Predicted_Winner'] = model.predict(race_df[features])
    race_df['Win_Probability'] = model.predict_proba(race_df[features])[:, 1]

    # Sort and show
    race_df = race_df.sort_values(by='Win_Probability', ascending=False)

    return render_template('result.html', race=selected_race, predictions=race_df)

if __name__ == '__main__':
    app.run(debug=True)
