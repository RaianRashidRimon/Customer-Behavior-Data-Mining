import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Hardcoded file path (raw string to handle backslashes)
file_path = r"D:\Academics\Semesters\4-2\Lab\Data Mining and Big Data Analysis Laboratory\Experiment Reports\Support Dataset\customer_data.csv"

# Read the CSV file
print("Loading data from:", file_path)
data = pd.read_csv(file_path)

# Preview the data
print("Sample data:\n", data.head())

# Set the target column
target_col = 'Will_purchase'

# Split into features and target
X = data.drop(columns=[target_col])
y = data[target_col]

# Convert categorical features into numeric using one-hot encoding
X = pd.get_dummies(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the decision tree model (ID3-style)
clf = DecisionTreeClassifier(criterion='entropy', max_depth=5)
clf.fit(X_train, y_train)

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=clf.classes_.astype(str), filled=True)

# Save the plot as a JPG image
output_filename = "decision_tree.jpg"
plt.savefig(output_filename, format='jpg', dpi=300)
plt.close()

print(f"✅ Decision tree saved as {output_filename}")
