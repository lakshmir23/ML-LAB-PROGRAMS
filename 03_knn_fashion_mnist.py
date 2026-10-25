from tensorflow.keras.datasets import fashion_mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train.reshape(len(X_train), -1) / 255.0
X_test = X_test.reshape(len(X_test), -1) / 255.0

k_values = [1, 3, 5]

for k in k_values:
    print(f"\n===== K = {k} =====")
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_train[:10000], y_train[:10000])

    y_pred = knn.predict(X_test[:2000])
    acc = accuracy_score(y_test[:2000], y_pred)

    print("Accuracy:", acc)
    print(classification_report(y_test[:2000], y_pred, zero_division=0))
