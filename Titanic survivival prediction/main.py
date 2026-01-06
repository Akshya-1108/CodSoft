# Import necessary libraries
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the Titanic dataset
titanic_data = pd.read_csv(r"Titanic survivival prediction\titanic_train.csv")

# Feature Engineering: Create the 'Along' column
titanic_data['Along'] = titanic_data['SibSp'] + titanic_data['Parch']
titanic_data['Along'] = (titanic_data['Along'] > 0).astype(int)  # Binary encoding: 1 if > 0, else 0

# Drop irrelevant columns
refined_titanic_data = titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked', 'SibSp', 'Parch'])

# Calculate average age for each passenger class (Pclass)
average_ages = refined_titanic_data.groupby('Pclass')['Age'].mean()

# Define a function to fill missing 'Age' values
def fill_age(row):
    age, pclass = row['Age'], row['Pclass']
    return average_ages[pclass] if pd.isnull(age) else age

# Apply the function to fill missing ages
refined_titanic_data['Age'] = refined_titanic_data.apply(fill_age, axis=1)

# Encode the 'Sex' column to numeric values
le = LabelEncoder()
refined_titanic_data['Sex'] = le.fit_transform(refined_titanic_data['Sex'])

# Uncomment below lines to visualize data
# sns.countplot(data=titanic_data, x='Pclass', hue='Sex')
# sns.displot(data=titanic_data, x='Age', bins=20)
# sns.boxplot(data=refined_titanic_data, x='Pclass', y='Age', palette="viridis")
# plt.figure(figsize=(10, 8))
sns.heatmap(refined_titanic_data.corr(), annot=True)
plt.show()

# Prepare data for modeling
X = refined_titanic_data.drop(columns=['Survived'])  # Features
y = refined_titanic_data['Survived']                # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Display dataset shapes for debugging
print(f"Shapes: X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

# Initialize and train the Decision Tree model
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_split=3)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
training_accuracy = model.score(X_train, y_train)
testing_accuracy = model.score(X_test, y_test)

# Print evaluation metrics
print(f"Training Accuracy: {training_accuracy:.2f}")
print(f"Testing Accuracy: {testing_accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualizing the tree
plt.figure(figsize=(15,15))
tree.plot_tree(model,feature_names=X.columns, class_names=['Dead','Survived'])
plt.show()
