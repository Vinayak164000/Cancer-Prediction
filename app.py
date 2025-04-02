from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline  


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template('home.html')
    
    else:
        data = CustomData(
            Age = int(request.form.get("Age", 0)),
            Number_of_sexual_partners = int(request.form.get("Number_of_sexual_partners", 0)),
            First_sexual_intercourse = int(request.form.get("First_sexual_intercourse", 0)),
            Num_of_pregnancies = int(request.form.get("Num_of_pregnancies", 0)),
            Smokes = int(request.form.get("Smokes", 0)),
            Smokes_years = float(request.form.get("Smokes_years", 0)),  # Use float for continuous values
            Smokes_packs_year = float(request.form.get("Smokes_packs_year", 0)),  
            Hormonal_Contraceptives = int(request.form.get("Hormonal_Contraceptives", 0)),
            STDs_Number_of_diagnosis = int(request.form.get("STDs_Number_of_diagnosis", 0)),
            Dx_Cancer = int(request.form.get("Dx_Cancer", 0)),
            Dx_CIN = int(request.form.get("Dx_CIN", 0)),
            Dx_HPV = int(request.form.get("Dx_HPV", 0)),
            Dx = int(request.form.get("Dx", 0)),
            Hinselmann = int(request.form.get("Hinselmann", 0)),
            Schiller = int(request.form.get("Schiller", 0)),
            Citology = int(request.form.get("Citology", 0))
)


        pred_df = data.get_data_as_data_frame()
        predict = PredictPipeline()
        pred_array = pred_df.to_numpy()
        results = predict.predict(pred_array)
        return render_template('home.html', results = results)
    
if __name__ == "__main__":
    app.run(debug= True)
