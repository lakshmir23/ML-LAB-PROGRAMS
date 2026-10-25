import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

df = pd.read_excel("Online Retail.xlsx")
df = df.dropna(subset=["CustomerID"])
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
df["TotalSpend"] = df["Quantity"] * df["UnitPrice"]

customer_df = df.groupby("CustomerID").agg({
    "Quantity": "sum",
    "UnitPrice": "mean",
    "TotalSpend": "sum",
    "Country": "first"
}).reset_index()

threshold = customer_df["TotalSpend"].median()
customer_df["HighValue"] = (customer_df["TotalSpend"] > threshold).astype(int)

encoder = LabelEncoder()
customer_df["Country"] = encoder.fit_transform(customer_df["Country"])

X = customer_df[["Quantity", "UnitPrice", "Country"]]
y = customer_df["HighValue"]

model = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
model.fit(X, y)

plt.figure(figsize=(20, 8))
plot_tree(
    model,
    feature_names=list(X.columns),
    class_names=["Low", "High"],
    filled=True
)
plt.title("Customer Segmentation using Decision Tree (ID3-style entropy)")
plt.tight_layout()
plt.show()

importance = pd.Series(model.feature_importances_, index=X.columns)
print("\nFeature Importance:\n", importance.sort_values(ascending=False))
