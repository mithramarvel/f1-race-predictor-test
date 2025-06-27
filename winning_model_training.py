# winner_model_training.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib  # for saving model

# === Step 1: Load the combined dataset ===
df = pd.read_csv('f1_2025_quali_data.csv')

# === Step 2: Basic cleaning and selecting relevant columns ===
# Make sure you have these columns after merge:
# ['Driver', 'Team_Race', 'Q1_sec', 'Q2_sec', 'Q3_sec', 'Grid_Pos', 'Race', 'Pos']

# Drop rows with missing data (you can later improve this with imputation)
df = df.dropna(subset=['Driver','Team', 'Q1_sec', 'Q2_sec', 'Q3_sec', 'Grid_Pos'])

# === Step 3: Create features ===
df['Avg_Quali_Time'] = df[['Q1_sec', 'Q2_sec', 'Q3_sec']].mean(axis=1)

# Ensure all teams are present before fitting
all_teams = set([
    'Red Bull Racing-Honda RBPT', 'Ferrari', 'Mercedes', 'McLaren-Mercedes',
    'Aston Martin Aramco-Mercedes', 'RB-Honda RBPT', 'Kick Sauber-Ferrari',
    'Williams-Mercedes', 'Alpine-Renault', 'Haas-Ferrari'
])

current_teams = set(df['Team'].unique())
missing_teams = all_teams - current_teams
dummy_row =[]
# Add dummy rows to include the missing teams
for team in missing_teams:
    dummy_row.append({
        'Avg_Quali_Time': df['Avg_Quali_Time'].mean(),
        'Grid_Pos': 10,
        'Team': team,
        'Driver': 'Dummy Driver',
        'Race': 'Dummy Grand Prix',
        'Winner': 0
    })
# === Make sure all 2025 drivers are present ===
all_drivers = [
    'Max Verstappen', 'Sergio Pérez', 'Charles Leclerc', 'Jack Doohan',
    'Lewis Hamilton', 'George Russell', 'Lando Norris', 'Oscar Piastri',
    'Fernando Alonso', 'Lance Stroll', 'Yuki Tsunoda', 'Gabriel Bortoleto',
    'Franco Colapinto', 'Zhou Guanyu', 'Kevin Magnussen', 'Nico Hülkenberg',
    'Alexander Albon', 'Logan Sargeant', 'Esteban Ocon', 'Pierre Gasly'
]

current_drivers = set(df['Driver'].unique())
missing_drivers = set(all_drivers) - current_drivers

for driver in missing_drivers:
    dummy_row.append ({
        'Avg_Quali_Time': df['Avg_Quali_Time'].mean(),
        'Grid_Pos': 10,
        'Team': 'Dummy Team',  # This will be encoded anyway
        'Driver': driver,
        'Race': 'Dummy Grand Prix',
        'Winner': 0
    })

df = pd.concat([df, pd.DataFrame(dummy_row)], ignore_index=True)

print("Unique teams in training data:")
print(df['Team'].dropna().unique())

print("Drivers in training data:")
print(sorted(df['Driver'].dropna().unique()))
# Encode categorical features
le_driver = LabelEncoder()
le_team = LabelEncoder()
le_race = LabelEncoder()

df['Driver_encoded'] = le_driver.fit_transform(df['Driver'])
le_team.fit(df['Team'])
df['Team_encoded'] = le_team.transform(df['Team'])
df['Race_encoded'] = le_race.fit_transform(df['Race'])

# Convert Grid_Pos and Pos to int (if they're not already)
df['Grid_Pos'] = pd.to_numeric(df['Grid_Pos'], errors='coerce')
#df['Pos'] = pd.to_numeric(df['Pos'], errors='coerce')
df = df.dropna(subset=['Grid_Pos'])   # Remove rows that failed to convert

# === Step 4: Define target ===
# Winner = 1 if finished 1st, else 0
df.loc[:,'Winner'] = (df['Grid_Pos'] == 1).astype(int)
df = df[df['Driver'] != 'Dummy Driver']
df = df[df['Team'] != 'Dummy Team']
# === Step 5: Select final features and target ===
features = ['Avg_Quali_Time', 'Grid_Pos', 'Team_encoded', 'Driver_encoded', 'Race_encoded']
# Drop original columns not needed for modeling
df = df.drop(columns=[
    'Driver', 'Team', 'Race',  # original string labels (now encoded)
    'Q1', 'Q2', 'Q3',          # raw quali time strings
    'Q1_sec', 'Q2_sec', 'Q3_sec',  # already averaged into Avg_Quali_Time
    'pos'                    # raw position — we're using it to create 'Winner'
], errors='ignore')  # errors='ignore' in case some columns are already gone

X = df[features]
y = df['Winner']

# === Step 6: Split dataset ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 7: Train model ===
model = RandomForestClassifier(n_estimators=100, class_weight ='balanced', random_state=42)
model.fit(X_train, y_train)

# === Step 8: Evaluate ===
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# === Step 9: Save model (optional) ===
joblib.dump(model, 'models/winner_predictor.pkl')
joblib.dump(le_driver, 'models/driver_encoder.pkl')
joblib.dump(le_team, 'models/team_encoder.pkl')
joblib.dump(le_race, 'models/race_encoder.pkl')

print("✅ Model and encoders saved!")
