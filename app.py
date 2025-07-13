#import all the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
#data handeling and reading and plotting graph  
df = pd.read_csv(r"D:\AI learn\heart-failure-prediction\heart.csv")
print(df.head())  
plt.figure(figsize=(10,8))
numeric_df = df.select_dtypes(include='number') 
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Class Distribution")
plt.xlabel("Presence of Heart Disease")
plt.ylabel("Count")
plt.show()
print(df.columns.tolist())
#selecting columns for features and target
features=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'ST_Slope','Oldpeak']
target=['HeartDisease']
#Dropping all the missing Values 
df=df[features +target].dropna()
#now labeling all the features & targets as as x,y
x= df[features]
y=df[target]
#splitting the data into test and train
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=42)
#seprate columns into numericals and categorial
number_fearures = ['Age','RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
catagorial_features=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina','ST_Slope']
#create columntransformers
preprocessor= ColumnTransformer(transformers=[
    ( 'num',StandardScaler(), number_fearures),
    ('cat',OneHotEncoder(),catagorial_features)
])
#building pipelines
model_pipeline= Pipeline([
    ("preprocesses", preprocessor),
    ("classifier", XGBClassifier(eval_metric='logloss'))
])
#fitting the pipline model
model_pipeline.fit(x_train,y_train)
#predict
y_pred= model_pipeline.predict(x_test)
#finding accuracy
print(accuracy_score(y_test,y_pred)) 
#setting parameter for search algorithm
param_dist = {
    'classifier__n_estimators': randint(100, 300),
    'classifier__max_depth': randint(3, 10),
    'classifier__learning_rate': uniform(0.01, 0.3),
    'classifier__subsample': uniform(0.5, 0.5),
    'classifier__colsample_bytree': uniform(0.5, 0.5)
}
#using grid search or randomize searchcv
random_search = RandomizedSearchCV(model_pipeline, param_distributions=param_dist, 
                                   n_iter=30, scoring='accuracy', cv=5, random_state=42)
random_search.fit(x_train, y_train)
#apply best values 
print("Best score:", random_search.best_score_)
print("Best parameters:", random_search.best_params_)
#saving the model 
import joblib
import streamlit as st
# Save the trained pipeline
joblib.dump(model_pipeline, "heart_disease_model_pipeline.pkl")
print("Model saved successfully")
#now runnig a web app
import streamlit
model = joblib.load("heart_disease_model_pipeline.pkl")
if __name__ == "__main__":
    st.title("Heart Disease predictor")
    model = joblib.load("heart_disease_model_pipeline.pkl")
    age = st.number_input("enter your age",min_value=1, max_value=120)
    sex= st.selectbox("select gender",["M","F"])
    CP= st.selectbox("Chest pain type",["ATA", "NAP", "ASY", "TA"])
    BP= st.number_input("enter BP")
    Colestrol= st.number_input("enter the colestrol")
    fbs = st.selectbox("Fasting BS", [0, 1])
    ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    maxhr = st.number_input("Max HR")
    Agnia = st.selectbox("do you feel chest pain ?",["Y","N"])
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
    oldpeak = st.number_input("Oldpeak")
if st.button("predict"):
    Data = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": CP,
        "RestingBP": BP,
        "Cholesterol": Colestrol,
        "FastingBS": fbs,
        "RestingECG": ecg,
        "MaxHR": maxhr,
        "ExerciseAngina": Agnia,
        "ST_Slope": st_slope,
        "Oldpeak": oldpeak
    }])

    
    prediction = model.predict(Data)
    result = "💔 Heart Disease Detected" if prediction[0] == 1 else "💚 No Heart Disease Detected"
    st.success(f"Prediction: {result}")
