import pickle
import os

import numpy as np
import pandas as pd

from flask import Flask
from flask import request
from flask import jsonify


model_file_name = os.environ.get('MODEL_FILE_NAME', 'best_model.bin')

with open(model_file_name, 'rb') as f_in:
    pipe = pickle.load(f_in)

app = Flask('medical_costs')


@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.get_json()
    input = pd.DataFrame(index=[0], data=json_data)
    y_pred = pipe.predict(input)

    result = {
        'charges': float(np.expm1(y_pred))
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
