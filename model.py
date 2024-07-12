import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Model Selection and Training
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# importing the data
df1 = pd.read_csv(r"C:\Users\ALVIN KAYI\Desktop\ml and ai\AI and ML\diabetes prediction model\Testing.csv")
df2 = pd.read_csv(r"C:\Users\ALVIN KAYI\Desktop\ml and ai\AI and ML\diabetes prediction model\Training.csv")

# concatenate the two dataframes to make the workload easier
df = pd.concat([df1,df2], axis=0)
print(df.head())
# check for the structure of the dataframe
print(df.info())

# check for null values in our case there is none
print(df.isnull().sum())


# Splitting data into features (X) and target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Standardizing features
scaler = StandardScaler()  # used so that it can normalize the values to between 0 - 1
# so that the sata is not,this can also be done in tensor flow.
X_scaled = scaler.fit_transform(X)

# Splitting into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Initialize Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model on the training data
model.fit(X_train, y_train)

# Predictions on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of Logistic Regression model: {accuracy:.2f}')

# Model Evaluation and Interpretation
from sklearn.metrics import classification_report, confusion_matrix

# Classification report and confusion matrix
print('Classification Report:')
print(classification_report(y_test, y_pred))

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print(df.columns)
