from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
df=pd.read_csv("diabetes.csv")
col=["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[col]=df[col].replace(0,np.nan)
df[col]=df[col].fillna(df[col].median())
x=df.drop("Outcome",axis=1)
y=df["Outcome"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
mymodel=LogisticRegression(class_weight="balanced",random_state=42,max_iter=2000)
mymodel.fit(x_train, y_train)
y_prob=mymodel.predict_proba(x_test)[:,1]
y_pred=(y_prob>=0.4).astype(int)
accuracy=accuracy_score(y_test,y_pred)
cf=confusion_matrix(y_test,y_pred)
print("Enter patient details:")
preg = float(input("Enter number of pregnancies: "))
glucose = float(input("Enter glucose level: "))
bp = float(input("Enter blood pressure: "))
skin = float(input("Enter skin thickness: "))
insulin = float(input("Enter insulin level: "))
bmi = float(input("Enter BMI: "))
dpf = float(input("Enter diabetes pedigree function: "))
age = float(input("Enter age: "))
new_data = pd.DataFrame(
    [[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
    columns=x.columns   
)
new_scaled=scaler.transform(new_data)
prediction=mymodel.predict(new_scaled)
prob=mymodel.predict_proba(new_scaled)
for i in range(len(prediction)):
  if prediction[i]==1:
    print("Person has diabetes")
  else:
    print("Doesn't have diabetes")
risk=prob[0][1]*100
print("Person has ",risk,"%"," risk")
print(accuracy)
