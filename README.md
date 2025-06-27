# 🏎️ F1 Race Winner Predictor with Machine Learning
#### Video Demo:  <https://youtu.be/kZLo-Q0z8AE>
#### Description:
Welcome to the **F1 Winner Predictor** — a machine learning-based project developed as my final submission for this CS50x course. This project uses real-time Formula 1 data of this year to predict the probability of each driver winning a chosen Grand Prix. The goal is to give the probability of a winning driver from analyzing qualifying performance, grid positions,and encoded data as well. This lite version of the prediction project brings together the topics i learned from this CS50x course such as python,sql,Flask,HTML,Bootstrap and so on.

Formula 1 races are one of my favourite passions and it's all about driver talent, team performance, qualifying position, and track circuits parameters. Predicting the winner is a challenging problem, often left to intuition or fan guesses. With this project i aim to use a my own dataset i scraped from sources such as wikipedia by building a classifier such as random forest which estimates the binary probability of each driver winning a race.

As there is insufficient data it won't be that accurate, but this is my first attempt on making my own dataset, train a model and predict the outcomes. To make it a bit user friendly i made a webpage using flask and the machine learning model was made using python and sql for the csv datasets.

---

## 📂 Project Structure

### `f1_2025_combined_data.csv`

A compiled dataset combining data of both qualifying and race classifications of this current 2025 season. It includes driver names, team names, grid positions, and qualifying session times (Q1, Q2, Q3) for each race. It is updated weekly with results scraped from reliable sources.

### `winner_model_training.py`

Responsible for training the machine learning model. It includes:

- Preprocessing steps like encoding drivers, teams, and races using LabelEncoder function.
- Feature engineering: average qualifying times,grid position,encoded team,driver and race.
- Training a `RandomForestClassifier` using `scikit-learn`.
- Saving the trained model and encoders by joblib module.

Output:

- `winner_predictor.pkl`
- `driver_encoder.pkl`
- `team_encoder.pkl`
- `race_encoder.pkl`

### `winner_predictor.py`

The script to predict winners of a given race:

- Loads the merged dataset and trained model.
- Accepts a race name as input.
- Cleans the dataset, encodes categorical columns, and calculates win probabilities.
- Prints a sorted table showing all drivers in the selected race.

### `app.py`

The Flask web application for user interaction:

- Lets users choose a race from a dropdown.
- Shows predicted winner probabilities in a clean table format.
- Integrates model logic with HTML rendering.

### `templates/results.html`

template for rendering prediction results on a webpage:

- Displays driver name, team, grid position, and win probability.
- Highlights the predicted winner with a 🏆 icon.

---

##  Key Features and Design Choices

### 1. Model Choice

Random Forest was chosen due to its simplicity, interpretability, and strong performance on tabular datasets. It also naturally handles categorical data and offers probability outputs.

### 2. Dummy Rows in Training

To prevent encoder mismatches during prediction, dummy entries for all current 2025 drivers and teams were added during training. This ensures that even if a new team/driver shows up in testing, the encoders will recognize them.

### 3. Win Probability Over Binary Prediction

Instead of outputting a single winner, the model gives a win probability for each driver. This reflects the uncertainty and complexity of races better than a binary result.

### 4. Frontend Design

Simple, clean layout built using HTML and Bootstrap. Winner is highlighted.

---

## Example Output

For the 2025 Miami Grand Prix, the output might look like:

| Driver         | Team     | Grid Pos | Avg Quali Time | Win Probability |
| -------------- | -------- | -------- | -------------- | --------------- |
| Max Verstappen | Red Bull | 1        | 86.57 s        | 33%             |
| Lando Norris   | McLaren  | 2        | 86.57 s        | 5%              |
| Oscar Piastri  | McLaren  | 4        | 86.55 s        | 2%              |

---

## How to Run

### 1. Install Dependencies
pip install flask pandas scikit-learn joblib
### 2. Train the model
python winner_model_training.py
### 3. To open the webpage
python app.py
