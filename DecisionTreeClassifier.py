from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

iris = load_iris()

tree = DecisionTreeClassifier()
tree.fit(iris.data, iris.target)

plt.figure(figsize=(10, 6))
plot_tree(tree, 
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True)
plt.title("Decision Tree Visualization")
plt.show()
