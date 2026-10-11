import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

wine=load_wine()
X,y=wine.data,wine.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

k_values =[1, 3, 5, 7, 9, ]

accuracy_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

plt.scatter(k_values, accuracy_scores)
plt.xlabel('k Values')
plt.ylabel('Accuracy')
plt.title('KNN Performance')
plt.show()

k_values= range(1,30)
accuracy_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    best_k = k_values[accuracy_scores.index(max(accuracy_scores))]
    print("Best value of k:", best_k)