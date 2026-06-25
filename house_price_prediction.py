import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load Dataset
data = pd.read_csv("dataset.csv")

print("Dataset:")
print(data.head())

# Features
X = data[['Area', 'Bedrooms']]

# Target
y = data['Price']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy Check
mae = mean_absolute_error(y_test, predictions)

print("\nMean Absolute Error:", mae)

# New House Prediction
area = 2400
bedrooms = 4

predicted_price = model.predict([[area, bedrooms]])

print("\nPredicted House Price:")
print(predicted_price[0])