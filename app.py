from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import joblib
import traceback
from flask_restful import reqparse
app = Flask(__name__)

@app.route("/", methods=['GET'])
def hello():
    return "hey"

@app.route('/predict', methods=['POST'])
def predict():
	lr = joblib.load("dopaNet.pkl")
	if lr:
		try:
			json = request.get_json()	 
			# print(json.get("context"))
			# list = [json.get("context"), json.get("question")]
			# # # model_columns = joblib.load("model_cols.pkl")
			# # temp=list(json["context"].values())
			# print(list)
			pred = {
				'context': json.get("context"),
				'question': json.get("question")
			}
			# # vals=np.array(list)
			prediction = lr.predict(pred)
			# print("here:",prediction)        
			# return jsonify({'prediction': str(prediction[0])})
			res = {
				'prediction': prediction['answer'],
				'score': prediction['score']
			}
			return (res)

		except:        
			# return jsonify({'trace': traceback.format_exc()})
			return ('no')
	else:
		return ('No model here to use')
    


if __name__ == '__main__':
    app.run()
    