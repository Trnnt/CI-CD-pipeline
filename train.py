import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib
df = pd.read_csv("D:\COA\CI-CD(MIOOPS)\Iris - Iris.csv")
df = df.drop("Id", axis=1)
x = df.drop("species", axis=1)
y = df["species"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
model = LogisticRegression()
model.fit(x_train, y_train)
pred = model.predict(x_test)
accuracy = accuracy_score(y_test, pred)
print(accuracy)
joblib.dump(model, "model.pkl")