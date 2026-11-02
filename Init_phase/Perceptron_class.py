from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
df = pd.read_csv(r"placement.csv")

df = df.iloc[:, 1:]
x = df.drop(columns="placement")
y = df["placement"]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)
model = Perceptron()

model.fit(xtrain, ytrain)

ypred = model.predict(xtest)

acc = accuracy_score(ytest, ypred)

print(acc)