import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=["GET","POST"])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        # get the user entered data
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )

        # convert it to dataframe
        df = data.get_input_as_dataframe()
        print(df)

        # predict the output
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(df)

        return render_template('home.html', results=round(results[0], 3))
    
if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True, port=8000)
    print("http://127.0.0.1:8000/")