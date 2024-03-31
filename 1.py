# Importing required libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from six import StringIO
import pydotplus
from IPython.display import Image

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Visualize the Decision Tree
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, rounded=True)
plt.show()

# Alternatively, you can visualize the Decision Tree using graphviz
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# Predict the class for new data
new_data = [[5.1, 3.5, 1.4, 0.2]]  # Example new data
predicted_class = clf.predict(new_data)
print(f"Predicted class for new data: {iris.target_names[predicted_class[0]]}")
