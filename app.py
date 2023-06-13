from flask import Flask, request
import joblib
app = Flask(__name__)

@app.route("/", methods=['GET'])
def hello():
    return "hey"

@app.route('/predict', methods=['POST'])
def predict():
	json = request.get_json()
	print(json)
	lr = joblib.load("dopaNet.pkl")
	if lr:
		try:
			json = request.get_json()
			
			# print(json)
			pred = {
				'context': json.get("context"),
				'question': json.get("question")
			}

			prediction = lr.predict(pred)

			res = {
				'prediction': prediction['answer'],
				'score': prediction['score']
			}
			
			return (res)

		except:
			return ('no')
	else:
		return ('No model here to use')
    


if __name__ == '__main__':
    app.run()
    