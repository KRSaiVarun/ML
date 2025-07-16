from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Get iris flower data
iris = load_iris()

# Make the decision tree model
tree = DecisionTreeClassifier()
tree.fit(iris.data, iris.target)

# Draw the tree
plt.figure(figsize=(10, 6))
plot_tree(tree, 
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True)
plt.show()
