#!/usr/bin/env python
# coding: utf-8

import numpy as np
from io import BytesIO
import urllib
from PIL import Image
import tflite_runtime.interpreter as tflite

from flask import Flask, request, jsonify


def download_image(url):
    with urllib.request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def resize_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size, Image.NEAREST)
    return image


def get_numpy_representation(image):
    np_image = np.array(image, dtype='float32')
    np_image = np.array([np_image])
    return np_image


def prepare_image(image_url, target_size):
    image = download_image(image_url)
    image = resize_image(image, target_size)
    np_image = get_numpy_representation(image)
    np_image /= 255.0
    return np_image


# load model
interpreter = tflite.Interpreter(model_path='./model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


flower_types_dict = {0: 'daisy', 1: 'dandelion', 2: 'rose', 3: 'sunflower', 4: 'tulip'}
target_size = (150, 150)


app = Flask('flower_type')


@app.route('/predict', methods=['POST'])
def predict():
    body = request.get_json()
    image_url = body['url']

    model_input = prepare_image(image_url, target_size)

    # make prediction
    interpreter.set_tensor(input_index, model_input)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)

    result = {
        'flower': flower_types_dict[pred.argmax()]
    }

    for idx, flower in flower_types_dict.items():
        result[f'{flower}_prob'] = float(pred[0][idx])

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
