import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

file_path = r"Enter the path of your CSV"
print("Loading data from:", file_path)
data = pd.read_csv(file_path)
print("Sample data:\n", data.head())
target_col = 'Will_purchase'
X = data.drop(columns=[target_col])
y = data[target_col]
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = DecisionTreeClassifier(criterion='entropy', max_depth=5)
clf.fit(X_train, y_train)
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=clf.classes_.astype(str), filled=True)
output_filename = "decision_tree.jpg"
plt.savefig(output_filename, format='jpg', dpi=300)
plt.close()
print(f"Decision tree saved as {output_filename}")
