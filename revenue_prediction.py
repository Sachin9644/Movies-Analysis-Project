# Movie Revenue Prediction Project  

 # Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

#  Load the dataset
file_path = "Top_10000_Movies.csv"
df = pd.read_csv(file_path, on_bad_lines='skip', encoding='utf-8', engine='python')

#  View basic information about the dataset
print("\nDataset Info:")
print(df.info())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values in each column:")
print(df.isnull().sum())

#  Select important columns for this project
cols_to_use = ['popularity', 'vote_average', 'vote_count', 'runtime', 'revenue']
df = df[cols_to_use]

# Drop rows with missing or zero revenue
df = df.dropna()
df = df[df['revenue'] > 0]

#  Summary statistics
print("\nSummary statistics:")
print(df.describe())

#  Data Visualization
plt.figure(figsize=(10, 5))
sns.histplot(df['revenue'], bins=50, kde=True, color='skyblue')
plt.title("Distribution of Revenue")
plt.xlabel("Revenue")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x=df['revenue'], color='lightgreen')
plt.title("Boxplot of Revenue")
plt.show()

plt.figure(figsize=(10, 5))
sns.scatterplot(x='popularity', y='revenue', data=df)
plt.title("Popularity vs Revenue")
plt.xlabel("Popularity")
plt.ylabel("Revenue")
plt.show()

plt.figure(figsize=(10, 5))
sns.scatterplot(x='vote_count', y='revenue', data=df)
plt.title("Vote Count vs Revenue")
plt.xlabel("Vote Count")
plt.ylabel("Revenue")
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x='vote_average', y='revenue', data=df.sort_values('vote_average'))
plt.xticks(rotation=90)
plt.title("Vote Average vs Revenue")
plt.xlabel("Vote Average")
plt.ylabel("Revenue")
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(df['runtime'], bins=30, kde=True, color='coral')
plt.title("Runtime Distribution")
plt.xlabel("Runtime")
plt.show()

px.scatter(df, x='runtime', y='revenue', title='Runtime vs Revenue').show()
px.histogram(df, x='vote_average', title='Vote Average Distribution').show()

#  Prepare data for machine learning
X = df[['popularity', 'vote_average', 'vote_count', 'runtime']]
y = df['revenue']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Train and evaluate models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5)
}

# Loop through models
for name, model in models.items():
    print(f"\nTraining: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared Score: {r2:.2f}")

    # Show prediction vs actual (only for first 50 samples)
    plt.figure(figsize=(10, 4))
    plt.plot(y_test.values[:50], label='Actual', marker='o')
    plt.plot(y_pred[:50], label='Predicted', marker='x')
    plt.title(f"Actual vs Predicted Revenue - {name}")
    plt.xlabel("Sample")
    plt.ylabel("Revenue")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Show coefficients for Linear Regression
    if name == 'Linear Regression':
        print("\nModel Coefficients:")
        for feature, coef in zip(X.columns, model.coef_):
            print(f"{feature}: {coef:.2f}")



