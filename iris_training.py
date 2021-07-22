from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz


iris = load_iris()
x = iris.data
y = iris.target

# training and testing data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= .5)

my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(x_train, y_train)

predictions = my_classifier.predict(x_test)
print(accuracy_score(y_test,predictions))


# visualization
# dot_data = tree.export_graphviz(my_classifier, out_file=None,
#                                 feature_names=iris.feature_names,
#                                 class_names=iris.target_names,
#                                 filled=True, rounded=True,
#                                 special_characters=True)
# graph = graphviz.Source(dot_data)
# graph.render("iris")
