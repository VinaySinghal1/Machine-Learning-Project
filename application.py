import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

## import ridge regressor and standard scaler pickel
rigde_model=pickle.load(open('moduls/Ridge.pkl','rb'))
standard_scaler=pickle.load(open('moduls/scaler.pkl','rb'))

application=Flask(__name__)
app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        print('step 1')
        Temperature = int(request.form.get('Temperature'))
        print('step 2')
        RH = float(request.form.get('RH'))
        print('step 3')
        Ws = float(request.form.get('Ws'))
        print('step 4')
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = request.form.get('Classes')  # Keep as string if it's categorical
        Region = request.form.get('Region')    # Keep as string if it's categorical

        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=rigde_model.predict(new_data_scaled)

        return render_template('home.html',results=result[0])

    else:
        return render_template('home.html')


if __name__=='__main__':
    app.run(host='0.0.0.0')

