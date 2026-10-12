from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5],
     [6], [7], [8], [9], [10]]

y = [7, 13, 31, 42, 54,
     69, 71, 83, 90, 95]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print("X_train:", X_train)
#print("X_test:", X_test)
#print("y_train:", y_train)
#print("y_test:", y_test)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Actual values:", y_test)
print("Predicted values:", y_pred)