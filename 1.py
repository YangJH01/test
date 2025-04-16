import pandas as pd

df = pd.read_csv("train.csv")
#print(df.head())

"""
print("\n정보요약")
print(df.info())
print(df.describe())
"""

#결측치 처리
df = df.drop(columns=["Name", "Ticket", "Cabin"])
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

#문자형 -> 숫자형, 모델은 문자열을 받지 못함.
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
embarked_map = {label: idx for idx, label in enumerate(df["Embarked"].unique())}
df["Embarked"] = df["Embarked"].map(embarked_map)

df.to_csv("train1.csv", index=False)

#-----------------------------------------------
df = pd.read_csv("train1.csv")

y = df["Survived"]
X = df.drop(columns=["Survived", "PassengerId"])

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_val, y_pred)
print(f"정확도 = {accuracy:.4f}")

