from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score, classification_report

categories = ["sci.space", "rec.sport.baseball", "comp.graphics"]

data = fetch_20newsgroups(
    subset="all",
    categories=categories,
    remove=("headers", "footers", "quotes")
)

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer_multi = CountVectorizer(stop_words="english", max_features=5000)
X_train_multi = vectorizer_multi.fit_transform(X_train)
X_test_multi = vectorizer_multi.transform(X_test)

vectorizer_bern = CountVectorizer(
    stop_words="english", binary=True, max_features=5000
)
X_train_bern = vectorizer_bern.fit_transform(X_train)
X_test_bern = vectorizer_bern.transform(X_test)

mnb = MultinomialNB()
mnb.fit(X_train_multi, y_train)

bnb = BernoulliNB()
bnb.fit(X_train_bern, y_train)

y_pred_mnb = mnb.predict(X_test_multi)
y_pred_bnb = bnb.predict(X_test_bern)

print("=== Multinomial NB ===")
print("Accuracy:", accuracy_score(y_test, y_pred_mnb))
print("F1 Score:", f1_score(y_test, y_pred_mnb, average="macro"))
print(classification_report(y_test, y_pred_mnb, zero_division=0))

print("\n=== Bernoulli NB ===")
print("Accuracy:", accuracy_score(y_test, y_pred_bnb))
print("F1 Score:", f1_score(y_test, y_pred_bnb, average="macro"))
print(classification_report(y_test, y_pred_bnb, zero_division=0))
