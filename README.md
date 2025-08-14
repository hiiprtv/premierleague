# Premier League Match Outcome Predictor

A machine learning project predicting football (soccer) match results (Home Win, Draw, Away Win) using historical Premier League data.

## Features
- Cleans and prepares match statistics (shots, fouls, corners, shots on target)
- Uses Random Forest Classifier for prediction
- Visualizes feature importance

## Tools Used
- Python (Pandas, Seaborn, Matplotlib)
- Scikit-learn (Random Forest)
- Data Source: [football-data.co.uk](https://www.football-data.co.uk)

## How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/football-outcome-predictor.git
   cd football-outcome-predictor
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download dataset and save as `data/epl_matches.csv`
4. Run predictor:
   ```bash
   python predictor.py
   ```

## Output
- Prints accuracy and classification report
- Shows feature importance plot

## Author
[Your Name] – aspiring Sports Data Analyst
