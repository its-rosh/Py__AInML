import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data = {
    'hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'attendance': [50, 60, 65, 70, 75, 80, 85, 90],
    'score': [35, 45, 50, 55, 60, 65, 70, 80]
}

df = pd.DataFrame(data)

X = df[['hours', 'attendance']]
y = df['score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("predictions",predictions)