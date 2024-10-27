from flask import Flask, request, jsonify
import pickle

# Load the DictVectorizer and model
with open('dv.bin', 'rb') as dv_file:
    dv = pickle.load(dv_file)

with open('model1.bin', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    X = dv.transform([client])
    probability = model.predict_proba(X)[0, 1]
    return jsonify({'probability': probability})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)

