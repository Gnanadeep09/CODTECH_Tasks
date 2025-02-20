from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt
data = load_iris()
x = data.data
y = data.target
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(x_train,y_train)
y_pred = dt_classifier.predict(x_test)
accuracy = (y_pred == y_test).mean()
print(f"accuracy: {accuracy:.2f}")
plt.figure(figsize=(10,8))
plot_tree(dt_classifier,filled=True,feature_names=data.feature_names,class_names=data.target_names)
plt.title=("Decison Tree Visualisation")
plt.show()  