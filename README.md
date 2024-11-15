import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Simulated dataset (replace with real-world network data)
data = {
    'Packet_Size': [500, 1500, 800, 200, 1000, 300, 50, 1400],
    'Duration': [2, 5, 1, 0.5, 3, 1, 0.2, 4],
    'Protocol': [1, 1, 0, 0, 1, 0, 0, 1],  # 1 = TCP, 0 = UDP
    'Flag': [1, 0, 0, 1, 1, 1, 0, 0],  # 1 = SYN, 0 = other
    'Malicious': [0, 1, 0, 1, 0, 0, 1, 1],  # 0 = normal, 1 = malicious
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Features (X) and Labels (y)
X = df[['Packet_Size', 'Duration', 'Protocol', 'Flag']]
y = df['Malicious']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Example: Detect a new threat
new_data = [[1200, 2.5, 1, 1]]  # Input features for a new packet
prediction = model.predict(new_data)
print("\nThreat Detection Result:")
print("Malicious" if prediction[0] == 1 else "Normal")
