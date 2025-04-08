import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
df = pd.read_csv(r"N:/LPU/Term 2/INT-524/CAs/Project/Model/multi_disease_dataset.csv")

# Print columns to verify (for debugging)
print("Columns in dataset:", df.columns)

# Define input and output columns (use correct lowercase names from dataset)
input_columns = ['age', 'bmi', 'bp', 'glucose', 'cholesterol', 'smoking', 'alcohol', 'physical_activity']
output_columns = ['diabetes', 'heart_disease', 'hypertension']

# Ensure all required columns are present
assert all(col in df.columns for col in input_columns + output_columns), "Missing columns in dataset"

# Drop rows with missing values
df = df.dropna()

# Split features and labels
X = df[input_columns]
y = df[output_columns]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'lifestyle_model.pkl')

print("✅ Model training completed and saved as 'multi_disease_model.pkl'")
