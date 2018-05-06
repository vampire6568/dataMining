import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus as dot

iris = load_iris()
test_index = [(i * 10) for i in range(15)]
sample_target = np.delete(iris.target, test_index)
sample_data = np.delete(iris.data, test_index, axis=0)
test_target = iris.target[test_index]
test_data = iris.data[test_index]

dtc = tree.DecisionTreeClassifier().fit(sample_data, sample_target)
decision_tree_out = StringIO()
tree.export_graphviz(dtc, rounded=True, special_characters=True, out_file=decision_tree_out,
                     feature_names=iris.feature_names, class_names=iris.target_names)
graph = dot.graph_from_dot_data(decision_tree_out.getvalue())
graph.write_pdf("decision_tree.pdf")
