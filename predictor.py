import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv("data/epl_matches.csv")

# Keep only relevant columns
df = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'FTR']]
df.dropna(inplace=True)

# Encode result: H = Home Win (1), D = Draw (0), A = Away Win (-1)
df['Result'] = df['FTR'].map({'H': 1, 'D': 0, 'A': -1})

# Features and Target
X = df[['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF']]
y = df['Result']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature importance
importances = model.feature_importances_
indices = importances.argsort()[::-1]
features = X.columns

plt.figure(figsize=(8, 5))
sns.barplot(x=importances[indices], y=features[indices])
plt.title("Feature Importance in Predicting Match Result")
plt.tight_layout()
plt.show()
