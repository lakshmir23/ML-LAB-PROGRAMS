from tensorflow.keras.datasets import fashion_mnist
from sklearn import svm
from sklearn.metrics import accuracy_score

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train[:5000] / 255.0
y_train = y_train[:5000]
X_test = X_test[:1000] / 255.0
y_test = y_test[:1000]

X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)

linear_model = svm.SVC(kernel="linear")
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

poly_model = svm.SVC(kernel="poly", degree=3)
poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)

rbf_model = svm.SVC(kernel="rbf")
rbf_model.fit(X_train, y_train)
y_pred_rbf = rbf_model.predict(X_test)

print("Linear SVM Accuracy:", accuracy_score(y_test, y_pred_linear))
print("Polynomial SVM Accuracy:", accuracy_score(y_test, y_pred_poly))
print("RBF SVM Accuracy:", accuracy_score(y_test, y_pred_rbf))
