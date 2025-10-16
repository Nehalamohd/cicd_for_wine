from flask import Flask,jsonify,request
import pickle
from prometheus_flask_exporter import PrometheusMetrics
import numpy as np
app=Flask(__name__)
metrics=PrometheusMetrics(app)
 
with open("model.pkl","rb") as f:
    model=pickle.load(f)


@app.route('/')
def home():
    return "model run successfully"

@app.route('/predicts',methods=['POST'])
def predicts():
    data=request.json
    feature=np.array(data['feature']).reshape(1,-1)
    prediction=model.predict(feature)[0]
    return jsonify({'prediction':int(prediction)})

@app.route('/health')
def health():
    return "Service is running"


if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000)