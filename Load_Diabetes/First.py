from sklearn.datasets import load_diabetes

# Load DataSet load_diabetes() 
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

# Split Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now Train The Modal
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predict The Modal
y_pred = regression.predict(X_test)

# Calculate Matrix 
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2_scor = r2_score(y_test, y_pred)

# Print All The Matrix 
print("Mean Squared Error Is : ", mse)
print()

print("Root Mean Squared Error Is : ", rmse)
print()

print("R2_score Is : ", r2_scor)


