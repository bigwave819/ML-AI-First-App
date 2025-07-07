# 📦 Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# 📥 Load dataset
data = pd.read_csv("student_scores.csv")

# 📊 Visualize data
plt.scatter(data['Hours'], data['Scores'], color='blue')
plt.title('Study Hours vs Score')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.grid(True)
plt.show()

# 🎯 Split data into input (X) and output (y)
X = data[['Hours']]   # Features
y = data['Scores']    # Target

# 📝 Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 Create linear regression model
model = LinearRegression()

# 📚 Train the model
model.fit(X_train, y_train)

# 🔍 Check the trained model's parameters
print(f"Intercept: {model.intercept_}")
print(f"Slope: {model.coef_[0]}")

# 📈 Plot regression line on scatter plot
line = model.coef_ * X + model.intercept_
plt.scatter(X, y)
plt.plot(X, line, color='red')  # Regression line
plt.title('Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.grid(True)
plt.show()

# 📊 Make predictions on test data
y_pred = model.predict(X_test)

# 📝 Compare actual vs predicted
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results)

# 📊 Evaluate the model
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
print("R2 Score:", metrics.r2_score(y_test, y_pred))

# 📌 Predict score for 9.5 hours of study
hours = [[9.5]]
predicted_score = model.predict(hours)
print(f"Predicted Score for 9.5 study hours = {predicted_score[0]:.2f}")
