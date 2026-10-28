import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
model = joblib.load("model.pkl")
test_df = pd.read_csv("D:\COA\CI-CD(MIOOPS)\Iris - Iris.csv")
test_df = test_df.drop("Id", axis = 1)
x = test_df.drop("Spcies", axis = 1)
y= test_df["Species"]

pred = model.predict(x)
accuracy = accuracy_score(pred,y)
print(accuracy)

assert accuracy>=0.98
assert len(pred) == len(y)
print("All test cases passed and model is ready to dispaly ")

