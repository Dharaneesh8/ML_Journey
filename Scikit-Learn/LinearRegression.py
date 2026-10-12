from sklearn.linear_model import LinearRegression
X=[[1],[3],[5],[7],[9]]
y=[35,50,66,79,93]

model=LinearRegression()
model.fit(X,y)

predictions=model.predict([[4]])
print("Predictions:", predictions)