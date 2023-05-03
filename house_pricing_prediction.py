import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load the housing dataset from CSV
df = pd.read_csv('Dataset/AmesHousing.csv')

# Split the dataset into features (X) and target (y)
X = df.drop(['SalePrice'], axis=1)
y = df['SalePrice']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values in the training set using mean imputation
numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns
numeric_transformer = SimpleImputer(strategy='mean')

categorical_features = X_train.select_dtypes(include=['object']).columns
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

X_train_imputed = preprocessor.fit_transform(X_train)

# Fit a linear regression model to the training data
model = LinearRegression()
model.fit(X_train_imputed, y_train)

# Impute missing values in the test set using mean imputation
X_test_imputed = preprocessor.transform(X_test)

# Use the model to make predictions on the test data
y_pred = model.predict(X_test_imputed)

# Calculate the R^2 score of the predictions
score = r2_score(y_test, y_pred)
print('R^2 score:', score)

# Create a scatter plot of the actual vs predicted sale prices
plt.scatter(y_test, y_pred)

# Add a diagonal line to represent perfect predictions
plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')

# Set the title and axis labels
plt.title('Actual vs Predicted Sale Prices')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')

# Display the plot
plt.show()
