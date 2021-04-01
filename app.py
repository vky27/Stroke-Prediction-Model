from flask import Flask, render_template, request
import joblib
import os
import numpy as np
import pickle

app= Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/result",methods=['POST','GET'])
def result():
    gender=int(request.form['gender'])
    age=int(request.form['age'])
    hypertension=int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    ever_married = int(request.form['ever_married'])
    work_type = int(request.form['work_type'])
    Residence_type = int(request.form['Residence_type'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    smoking_status = int(request.form['smoking_status'])

    if work_type == 0:
        Govt_job = 1
        Private = 0
        Self_Employed = 0
        Children = 0

    elif work_type == 1:
        Govt_job = 0
        Private = 0
        Self_Employed = 0
        Children = 0

    elif work_type == 2:
        Govt_job = 0
        Private = 1
        Self_Employed = 0
        Children = 0

    elif work_type == 3:
        Govt_job = 0
        Private = 0
        Self_Employed = 1
        Children = 0

    elif work_type == 4:
        Govt_job = 0
        Private = 0
        Self_Employed = 0
        Children = 1


    x=np.array([age,hypertension,heart_disease,avg_glucose_level,bmi,smoking_status,gender,ever_married,
                Govt_job,Private,Self_Employed,Children,Residence_type]).reshape(1,-1)

    
    scaler=pickle.load(open('scaler.pkl','rb'))

    x=scaler.transform(x)

    
    sv=joblib.load(open('sv.sav','rb'))

    Y_pred=sv.predict(x)

    # for No Stroke Risk
    if Y_pred==0:
        return render_template('nostroke.html')
    else:
        return render_template('stroke.html')

if __name__=="__main__":
    app.run(debug=True)
